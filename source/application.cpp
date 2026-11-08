#include "application.hpp"

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <fstream>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>

namespace application {
	namespace {

	struct Vertex {
		glm::vec3 pos;
		glm::vec3 normal;
	};

	struct PointLight {
		glm::vec4 position;
		glm::vec4 color;
	};

	struct SpotLight {
		glm::vec4 position;
		glm::vec4 direction;
		glm::vec4 color;
	};

	// Set 0: Общие данные сцены
	struct SceneUBO {
		glm::mat4 view;
		glm::mat4 proj;
		glm::vec4 cameraPos;
		glm::vec4 ambient;
		glm::vec4 dirLightDir;
		glm::vec4 dirLightColor;
		int numPointLights;
		int numSpotLights;
		float pad[2];
	};

	struct ModelUBO {
		glm::mat4 model;
		glm::mat4 normalMat;
		glm::vec4 albedo;
		glm::vec4 specColor;
		float shininess;
		float isEmissive;
		float pad[2];
	};

	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	VkPipeline graphics_pipeline = VK_NULL_HANDLE;
	VkPipeline line_pipeline = VK_NULL_HANDLE; // НОВЫЙ: конвейер для линий
	VkDescriptorSetLayout descriptor_set_layouts[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
	VkDescriptorSet descriptor_sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};

	VkBuffer vertex_buffer = VK_NULL_HANDLE;
	VmaAllocation vertex_buffer_allocation = VK_NULL_HANDLE;
	uint32_t vertex_count = 0;

	VkBuffer scene_ubo_buffer = VK_NULL_HANDLE;
	VmaAllocation scene_ubo_allocation = VK_NULL_HANDLE;
	void* scene_ubo_mapped = nullptr;

	VkBuffer model_ubo_buffer = VK_NULL_HANDLE;
	VmaAllocation model_ubo_allocation = VK_NULL_HANDLE;
	void* model_ubo_mapped = nullptr;

	VkBuffer point_light_buffer = VK_NULL_HANDLE;
	VmaAllocation point_light_allocation = VK_NULL_HANDLE;
	void* point_light_mapped = nullptr;

	VkBuffer spot_light_buffer = VK_NULL_HANDLE;
	VmaAllocation spot_light_allocation = VK_NULL_HANDLE;
	void* spot_light_mapped = nullptr;

	VkBuffer cone_line_buffer = VK_NULL_HANDLE;
	VmaAllocation cone_line_allocation = VK_NULL_HANDLE;
	uint32_t cone_line_vertex_count = 0;

	glm::vec3 camera_pos = {0.0f, 2.0f, 8.0f};
	float camera_pitch = -10.0f;
	float camera_yaw = -90.0f;
	float camera_fov = 45.0f;

	SceneUBO scene_ubo_data{};

	std::vector<PointLight> point_lights;
	std::vector<SpotLight> spot_lights;

	glm::vec3 albedo = {0.8f, 0.3f, 0.3f};
	glm::vec3 specular_color = {1.0f, 1.0f, 1.0f};
	float shininess = 32.0f;

	bool show_point_lights = true;
	bool show_spot_lights = true;

	VkBuffer light_cube_buffer = VK_NULL_HANDLE;
	VmaAllocation light_cube_allocation = VK_NULL_HANDLE;
	uint32_t light_cube_vertex_count = 0;

	std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);
		if (!file.is_open()) {
			std::cerr << "Failed to open file: " << filename << '\n';
			return {};
		}
		size_t file_size = static_cast<size_t>(file.tellg());
		std::vector<char> buffer(file_size);
		file.seekg(0);
		file.read(buffer.data(), file_size);
		return buffer;
	}

	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo create_info{};
		create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		create_info.codeSize = code.size();
		create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());
		VkShaderModule module = VK_NULL_HANDLE;
		vkCreateShaderModule(graphics::internal::context.device, &create_info, nullptr, &module);
		return module;
	}

	void createVertexBuffer() {
		std::vector<glm::vec3> base_pos = {
			{ 1.0f,      1.0f/3.0f,  1.0f/3.0f}, { 1.0f/3.0f, 1.0f,      1.0f/3.0f},
			{ 1.0f/3.0f, 1.0f/3.0f, 1.0f     }, { 1.0f,     -1.0f/3.0f, -1.0f/3.0f},
			{ 1.0f/3.0f,-1.0f/3.0f, -1.0f    }, { 1.0f/3.0f,-1.0f,     -1.0f/3.0f},
			{-1.0f/3.0f, 1.0f,     -1.0f/3.0f}, {-1.0f/3.0f, 1.0f/3.0f,-1.0f     },
			{-1.0f,      1.0f/3.0f,-1.0f/3.0f}, {-1.0f/3.0f,-1.0f/3.0f, 1.0f     },
			{-1.0f/3.0f,-1.0f,      1.0f/3.0f}, {-1.0f,     -1.0f/3.0f, 1.0f/3.0f}
		};
		std::vector<uint32_t> indices = {
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
			0, 3, 4,  0, 4, 7,  0, 7, 6,  0, 6, 1,
			0, 3, 5,  0, 5, 10, 0, 10, 9, 0, 9, 2,
			1, 6, 8,  1, 8, 11, 1, 11, 9, 1, 9, 2,
			4, 7, 8,  4, 8, 11, 4, 11, 10, 4, 10, 5,
		};

		std::vector<Vertex> vertices;
		for (size_t i = 0; i < indices.size(); i += 3) {
			glm::vec3 p0 = base_pos[indices[i]];
			glm::vec3 p1 = base_pos[indices[i+1]];
			glm::vec3 p2 = base_pos[indices[i+2]];
			
			glm::vec3 normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
			
			vertices.push_back({p0, normal});
			vertices.push_back({p1, normal});
			vertices.push_back({p2, normal});
		}
		vertex_count = static_cast<uint32_t>(vertices.size());

		VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer_info.size = sizeof(Vertex) * vertices.size();
		buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		VmaAllocationCreateInfo alloc_info{};
		alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
		alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo alloc_result{};
		vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &alloc_info, &vertex_buffer, &vertex_buffer_allocation, &alloc_result);
		std::memcpy(alloc_result.pMappedData, vertices.data(), buffer_info.size);
	}

	void createLightCubeBuffer() {
		std::vector<Vertex> vertices = {
			{{ 0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}}, {{ 0.5f,  0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}}, {{ 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}},
			{{ 0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}}, {{ 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}}, {{ 0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}},
			{{-0.5f, -0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}}, {{-0.5f,  0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}}, {{-0.5f,  0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}},
			{{-0.5f, -0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}}, {{-0.5f,  0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}}, {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}},
			{{-0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}}, {{ 0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}}, {{ 0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}},
			{{-0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}}, {{ 0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}}, {{-0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}},
			{{-0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}}, {{ 0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}}, {{ 0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}},
			{{-0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}}, {{ 0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}}, {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}},
			{{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}}, {{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}}, {{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
			{{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}}, {{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}}, {{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
			{{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}}, {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}}, {{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
			{{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}}, {{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}}, {{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
		};
		light_cube_vertex_count = static_cast<uint32_t>(vertices.size());

		VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer_info.size = sizeof(Vertex) * vertices.size();
		buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		VmaAllocationCreateInfo alloc_info{};
		alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
		alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo alloc_result{};
		vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &alloc_info, &light_cube_buffer, &light_cube_allocation, &alloc_result);
		std::memcpy(alloc_result.pMappedData, vertices.data(), buffer_info.size);
	}

	void createConeLineBuffer() {
		std::vector<Vertex> vertices;
		int segments = 16;
		float radius = 1.0f;
		float height = 2.0f;
		
		glm::vec3 tip(0.0f, 0.0f, 0.0f);
		
		for (int i = 0; i < segments; ++i) {
			float a1 = (float)i / segments * 2.0f * 3.14159f;
			float a2 = (float)(i + 1) / segments * 2.0f * 3.14159f;
			
			glm::vec3 p1(radius * cos(a1), radius * sin(a1), -height);
			glm::vec3 p2(radius * cos(a2), radius * sin(a2), -height);
			
			vertices.push_back({tip, {0,0,1}});
			vertices.push_back({p1, {0,0,1}});
			vertices.push_back({p1, {0,0,1}});
			vertices.push_back({p2, {0,0,1}});
		}
		cone_line_vertex_count = static_cast<uint32_t>(vertices.size());

		VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer_info.size = sizeof(Vertex) * vertices.size();
		buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		VmaAllocationCreateInfo alloc_info{};
		alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
		alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo alloc_result{};
		vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &alloc_info, &cone_line_buffer, &cone_line_allocation, &alloc_result);
		std::memcpy(alloc_result.pMappedData, vertices.data(), buffer_info.size);
	}

	void createMappedBuffer(VkBuffer& buffer, VmaAllocation& alloc, void*& mapped, size_t size, VkBufferUsageFlags usage) {
		VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer_info.size = size;
		buffer_info.usage = usage;
		buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo alloc_info{};
		alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
		alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

		VmaAllocationInfo alloc_result{};
		vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &alloc_info, &buffer, &alloc, &alloc_result);
		mapped = alloc_result.pMappedData;
	}

	void createDescriptorSets() {
		VkDescriptorSetLayoutBinding bindings0[3]{};
		bindings0[0].binding = 0; bindings0[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; bindings0[0].descriptorCount = 1; bindings0[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		bindings0[1].binding = 1; bindings0[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bindings0[1].descriptorCount = 1; bindings0[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		bindings0[2].binding = 2; bindings0[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bindings0[2].descriptorCount = 1; bindings0[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layout_info0{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		layout_info0.bindingCount = 3; layout_info0.pBindings = bindings0;
		vkCreateDescriptorSetLayout(graphics::internal::context.device, &layout_info0, nullptr, &descriptor_set_layouts[0]);

		VkDescriptorSetLayoutBinding binding1{};
		binding1.binding = 0; 
		binding1.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
		binding1.descriptorCount = 1; 
		binding1.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layout_info1{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		layout_info1.bindingCount = 1; layout_info1.pBindings = &binding1;
		vkCreateDescriptorSetLayout(graphics::internal::context.device, &layout_info1, nullptr, &descriptor_set_layouts[1]);

		VkDescriptorPoolSize pool_sizes[] = {
			{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
			{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}
		};
		VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		pool_info.poolSizeCount = 3; pool_info.pPoolSizes = pool_sizes; pool_info.maxSets = 2;
		vkCreateDescriptorPool(graphics::internal::context.device, &pool_info, nullptr, &descriptor_pool);

		VkDescriptorSetAllocateInfo alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		alloc_info.descriptorPool = descriptor_pool; alloc_info.descriptorSetCount = 2; alloc_info.pSetLayouts = descriptor_set_layouts;
		vkAllocateDescriptorSets(graphics::internal::context.device, &alloc_info, descriptor_sets);

		VkDescriptorBufferInfo scene_info{scene_ubo_buffer, 0, sizeof(SceneUBO)};
		VkDescriptorBufferInfo point_info{point_light_buffer, 0, VK_WHOLE_SIZE};
		VkDescriptorBufferInfo spot_info{spot_light_buffer, 0, VK_WHOLE_SIZE};

		VkWriteDescriptorSet writes0[3]{};
		writes0[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; writes0[0].dstSet = descriptor_sets[0]; writes0[0].dstBinding = 0; writes0[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes0[0].descriptorCount = 1; writes0[0].pBufferInfo = &scene_info;
		writes0[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; writes0[1].dstSet = descriptor_sets[0]; writes0[1].dstBinding = 1; writes0[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes0[1].descriptorCount = 1; writes0[1].pBufferInfo = &point_info;
		writes0[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; writes0[2].dstSet = descriptor_sets[0]; writes0[2].dstBinding = 2; writes0[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes0[2].descriptorCount = 1; writes0[2].pBufferInfo = &spot_info;
		vkUpdateDescriptorSets(graphics::internal::context.device, 3, writes0, 0, nullptr);

		VkDescriptorBufferInfo model_info{model_ubo_buffer, 0, sizeof(ModelUBO)};
		VkWriteDescriptorSet write1{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		write1.dstSet = descriptor_sets[1]; write1.dstBinding = 0; write1.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; write1.descriptorCount = 1; write1.pBufferInfo = &model_info;
		vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write1, 0, nullptr);
	}

	void createGraphicsPipeline() {
		auto vert_code = readFile("shaders/shader.vert.spv");
		auto frag_code = readFile("shaders/shader.frag.spv");
		if (vert_code.empty() || frag_code.empty()) {
			std::cerr << "Failed to load shaders.\n";
			return;
		}

		VkShaderModule vert_module = createShaderModule(vert_code);
		VkShaderModule frag_module = createShaderModule(frag_code);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vert_module; stages[0].pName = "main";
		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = frag_module; stages[1].pName = "main";

		VkVertexInputBindingDescription binding{};
		binding.binding = 0; binding.stride = sizeof(Vertex); binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription attr_pos{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
		VkVertexInputAttributeDescription attr_norm{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
		VkVertexInputAttributeDescription attributes[] = { attr_pos, attr_norm };

		VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
		vertex_input.vertexBindingDescriptionCount = 1; vertex_input.pVertexBindingDescriptions = &binding;
		vertex_input.vertexAttributeDescriptionCount = 2; vertex_input.pVertexAttributeDescriptions = attributes;

		VkPipelineViewportStateCreateInfo viewport_state{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
		viewport_state.viewportCount = 1; viewport_state.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL; rasterizer.lineWidth = 1.0f; rasterizer.cullMode = VK_CULL_MODE_NONE; rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

		VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineDepthStencilStateCreateInfo depth_stencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
		depth_stencil.depthTestEnable = VK_TRUE; depth_stencil.depthWriteEnable = VK_TRUE; depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;

		VkPipelineColorBlendAttachmentState blend_attachment{};
		blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT; blend_attachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo color_blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
		color_blending.attachmentCount = 1; color_blending.pAttachments = &blend_attachment;

		VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamic_state{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
		dynamic_state.dynamicStateCount = 2; dynamic_state.pDynamicStates = dynamic_states;

		VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		layout_info.setLayoutCount = 2; layout_info.pSetLayouts = descriptor_set_layouts;
		vkCreatePipelineLayout(graphics::internal::context.device, &layout_info, nullptr, &pipeline_layout);

		VkPipelineInputAssemblyStateCreateInfo input_assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
		input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		VkGraphicsPipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
		pipeline_info.stageCount = 2; pipeline_info.pStages = stages;
		pipeline_info.pVertexInputState = &vertex_input; pipeline_info.pInputAssemblyState = &input_assembly;
		pipeline_info.pViewportState = &viewport_state; pipeline_info.pRasterizationState = &rasterizer;
		pipeline_info.pMultisampleState = &multisampling; pipeline_info.pDepthStencilState = &depth_stencil;
		pipeline_info.pColorBlendState = &color_blending; pipeline_info.pDynamicState = &dynamic_state;
		pipeline_info.layout = pipeline_layout; pipeline_info.renderPass = graphics::internal::context.render_pass; pipeline_info.subpass = 0;
		
		vkCreateGraphicsPipelines(graphics::internal::context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline);

		VkPipelineInputAssemblyStateCreateInfo line_input_assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
		line_input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
		
		pipeline_info.pInputAssemblyState = &line_input_assembly;
		vkCreateGraphicsPipelines(graphics::internal::context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &line_pipeline);

		vkDestroyShaderModule(graphics::internal::context.device, vert_module, nullptr);
		vkDestroyShaderModule(graphics::internal::context.device, frag_module, nullptr);
	}

	}

	bool initialize() {
		createVertexBuffer();
		createLightCubeBuffer();
		createConeLineBuffer();
		
		createMappedBuffer(scene_ubo_buffer, scene_ubo_allocation, scene_ubo_mapped, sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
		
		size_t min_alignment = 256; 
		size_t aligned_size = (sizeof(ModelUBO) + min_alignment - 1) & ~(min_alignment - 1);
		createMappedBuffer(model_ubo_buffer, model_ubo_allocation, model_ubo_mapped, aligned_size * 100, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
		
		createMappedBuffer(point_light_buffer, point_light_allocation, point_light_mapped, sizeof(PointLight) * 100, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		createMappedBuffer(spot_light_buffer, spot_light_allocation, spot_light_mapped, sizeof(SpotLight) * 100, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		
		createDescriptorSets();
		createGraphicsPipeline();

		point_lights.push_back({{2.0f, 2.0f, 2.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}});
		spot_lights.push_back({{0.0f, 5.0f, 0.0f, 1.0f}, {0.0f, -1.0f, 0.0f, glm::cos(glm::radians(15.0f))}, {0.0f, 1.0f, 0.0f, glm::cos(glm::radians(25.0f))}});

		scene_ubo_data.ambient = glm::vec4(0.15f, 0.15f, 0.15f, 1.0f);
		scene_ubo_data.dirLightDir = glm::vec4(-0.5f, -1.0f, -0.5f, 0.0f);
		scene_ubo_data.dirLightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

		return true;
	}

	void shutdown() {
		auto& context = graphics::internal::context;
		vkQueueWaitIdle(context.graphics_queue);

		if (graphics_pipeline) vkDestroyPipeline(context.device, graphics_pipeline, nullptr);
		if (line_pipeline) vkDestroyPipeline(context.device, line_pipeline, nullptr);
		if (pipeline_layout) vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
		if (descriptor_pool) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
		if (descriptor_set_layouts[0]) vkDestroyDescriptorSetLayout(context.device, descriptor_set_layouts[0], nullptr);
		if (descriptor_set_layouts[1]) vkDestroyDescriptorSetLayout(context.device, descriptor_set_layouts[1], nullptr);
		
		vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_buffer_allocation);
		vmaDestroyBuffer(context.allocator, light_cube_buffer, light_cube_allocation);
		vmaDestroyBuffer(context.allocator, cone_line_buffer, cone_line_allocation);
		vmaDestroyBuffer(context.allocator, scene_ubo_buffer, scene_ubo_allocation);
		vmaDestroyBuffer(context.allocator, model_ubo_buffer, model_ubo_allocation);
		vmaDestroyBuffer(context.allocator, point_light_buffer, point_light_allocation);
		vmaDestroyBuffer(context.allocator, spot_light_buffer, spot_light_allocation);
	}

	void update(double time) {
		ImGui::Begin("Scene Controls");
		
		if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::SliderFloat3("Position", glm::value_ptr(camera_pos), -10.0f, 10.0f);
			ImGui::SliderFloat("Yaw", &camera_yaw, -180.0f, 180.0f);
			ImGui::SliderFloat("Pitch", &camera_pitch, -89.0f, 89.0f);
			ImGui::SliderFloat("FOV", &camera_fov, 10.0f, 120.0f);
		}

		if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::ColorEdit3("Albedo", glm::value_ptr(albedo));
			ImGui::ColorEdit3("Specular", glm::value_ptr(specular_color));
			ImGui::SliderFloat("Shininess", &shininess, 2.0f, 256.0f);
		}

		if (ImGui::CollapsingHeader("Directional Light")) {
			ImGui::SliderFloat3("Dir Direction", glm::value_ptr(scene_ubo_data.dirLightDir), -1.0f, 1.0f);
			ImGui::ColorEdit3("Dir Color", glm::value_ptr(scene_ubo_data.dirLightColor));
			ImGui::ColorEdit3("Ambient", glm::value_ptr(scene_ubo_data.ambient));
		}

		if (ImGui::CollapsingHeader("Point Lights")) {
			if (ImGui::Button("Add Point Light") && point_lights.size() < 100) {
				point_lights.push_back({{0,2,0,1}, {1,1,1,1}});
			}
			for (size_t i = 0; i < point_lights.size(); ++i) {
				ImGui::PushID(i);
				ImGui::Text("Light %d", (int)i);
				ImGui::SliderFloat3("Pos", glm::value_ptr(point_lights[i].position), -10.0f, 10.0f);
				ImGui::ColorEdit3("Col", glm::value_ptr(point_lights[i].color));
				ImGui::SliderFloat("Intensity", &point_lights[i].position.w, 0.0f, 50.0f);
				if (ImGui::Button("Remove")) { point_lights.erase(point_lights.begin() + i); ImGui::PopID(); break; }
				ImGui::PopID();
			}
		}

		if (ImGui::CollapsingHeader("Spot Lights")) {
			if (ImGui::Button("Add Spot Light") && spot_lights.size() < 100) {
				spot_lights.push_back({{0,5,0,1}, {0,-1,0,glm::cos(glm::radians(15.0f))}, {1,1,1,glm::cos(glm::radians(25.0f))}});
			}
			for (size_t i = 0; i < spot_lights.size(); ++i) {
				ImGui::PushID(i + 100);
				ImGui::Text("Spot %d", (int)i);
				ImGui::SliderFloat3("Pos", glm::value_ptr(spot_lights[i].position), -10.0f, 10.0f);
				ImGui::SliderFloat3("Dir", glm::value_ptr(spot_lights[i].direction), -1.0f, 1.0f);
				
				ImGui::SliderFloat("Intensity", &spot_lights[i].position.w, 0.0f, 100.0f); 
				
				float inner = glm::degrees(glm::acos(spot_lights[i].direction.w));
				float outer = glm::degrees(glm::acos(spot_lights[i].color.w));
				
				if (ImGui::SliderFloat("Inner Angle", &inner, 1.0f, 45.0f)) 
					spot_lights[i].direction.w = glm::cos(glm::radians(inner));
				if (ImGui::SliderFloat("Outer Angle", &outer, 1.0f, 45.0f)) 
					spot_lights[i].color.w = glm::cos(glm::radians(outer));
					
				if (ImGui::Button("Remove")) { 
					spot_lights.erase(spot_lights.begin() + i); 
					ImGui::PopID(); 
					break; 
				}
				ImGui::PopID();
			}
		}

		ImGui::Checkbox("Show Point Lights", &show_point_lights);
		ImGui::Checkbox("Show Spot Lights", &show_spot_lights);
		ImGui::End();

		float yaw_rad = glm::radians(camera_yaw);
		float pitch_rad = glm::radians(camera_pitch);
		glm::vec3 front;
		front.x = cos(yaw_rad) * cos(pitch_rad);
		front.y = sin(pitch_rad);
		front.z = sin(yaw_rad) * cos(pitch_rad);

		scene_ubo_data.view = glm::lookAt(camera_pos, camera_pos + front, glm::vec3(0,1,0));
		float aspect = (float)graphics::internal::context.swapchain_extent.width / (float)graphics::internal::context.swapchain_extent.height;
		scene_ubo_data.proj = glm::perspective(glm::radians(camera_fov), aspect, 0.1f, 100.0f);
		scene_ubo_data.proj[1][1] *= -1.0f;
		scene_ubo_data.cameraPos = glm::vec4(camera_pos, 1.0f);
		scene_ubo_data.numPointLights = (int)point_lights.size();
		scene_ubo_data.numSpotLights = (int)spot_lights.size();

		std::memcpy(scene_ubo_mapped, &scene_ubo_data, sizeof(SceneUBO));
		if (!point_lights.empty()) std::memcpy(point_light_mapped, point_lights.data(), point_lights.size() * sizeof(PointLight));
		if (!spot_lights.empty()) std::memcpy(spot_light_mapped, spot_lights.data(), spot_lights.size() * sizeof(SpotLight));
	}

	void render(const graphics::internal::FrameData& fd) {
		vkResetCommandBuffer(fd.command_buffer, 0);
		VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(fd.command_buffer, &begin_info);

		VkClearValue clear_values[2];
		clear_values[0].color = {{0.05f, 0.05f, 0.05f, 1.0f}};
		clear_values[1].depthStencil = {1.0f, 0};

		VkRenderPassBeginInfo render_pass_info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
		render_pass_info.renderPass = graphics::internal::context.render_pass;
		render_pass_info.framebuffer = fd.framebuffer;
		render_pass_info.renderArea = { {0, 0}, graphics::internal::context.swapchain_extent };
		render_pass_info.clearValueCount = 2;
		render_pass_info.pClearValues = clear_values;

		vkCmdBeginRenderPass(fd.command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

		VkViewport viewport{0.0f, 0.0f, (float)graphics::internal::context.swapchain_extent.width, (float)graphics::internal::context.swapchain_extent.height, 0.0f, 1.0f};
		vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
		VkRect2D scissor{{0, 0}, graphics::internal::context.swapchain_extent};
		vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_sets[0], 0, nullptr);

		std::vector<ModelUBO> model_ubos;
		
		// 1. Тетраэдр
		ModelUBO tetra{};
		tetra.model = glm::mat4(1.0f);
		tetra.normalMat = glm::transpose(glm::inverse(tetra.model));
		tetra.albedo = glm::vec4(albedo, 1.0f);
		tetra.specColor = glm::vec4(specular_color, 1.0f);
		tetra.shininess = shininess;
		tetra.isEmissive = 0.0f;
		model_ubos.push_back(tetra);

		// 2. Кубики точечных источников
		if (show_point_lights) {
			for (const auto& pl : point_lights) {
				ModelUBO light{};
				light.model = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(pl.position)), glm::vec3(0.2f));
				light.normalMat = glm::transpose(glm::inverse(light.model));
				light.albedo = glm::vec4(pl.color.r, pl.color.g, pl.color.b, 1.0f);
				light.isEmissive = 1.0f;
				model_ubos.push_back(light);
			}
		}

		// 3. Кубики прожекторов
		if (show_spot_lights) {
			for (const auto& sl : spot_lights) {
				ModelUBO light{};
				light.model = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(sl.position)), glm::vec3(0.2f));
				light.albedo = glm::vec4(sl.color.r, sl.color.g, sl.color.b, 1.0f);
				light.isEmissive = 1.0f;
				model_ubos.push_back(light);
			}
		}

		size_t aligned_size = 256;
		for (size_t i = 0; i < model_ubos.size(); ++i) {
			std::memcpy((char*)model_ubo_mapped + (i * aligned_size), &model_ubos[i], sizeof(ModelUBO));
		}

		uint32_t dynamic_offset = 0;
		VkDeviceSize offsets[] = {0};

		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &descriptor_sets[1], 1, &dynamic_offset);
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, offsets);
		vkCmdDraw(fd.command_buffer, vertex_count, 1, 0, 0);
		dynamic_offset += aligned_size;

		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &light_cube_buffer, offsets);
		size_t light_count = model_ubos.size() - 1;
		for (size_t i = 0; i < light_count; ++i) {
			vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &descriptor_sets[1], 1, &dynamic_offset);
			vkCmdDraw(fd.command_buffer, light_cube_vertex_count, 1, 0, 0);
			dynamic_offset += aligned_size;
		}

		if (show_spot_lights && !spot_lights.empty()) {
			vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);
			vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &cone_line_buffer, offsets);

			for (const auto& sl : spot_lights) {
				ModelUBO cone{};
				glm::mat4 m = glm::translate(glm::mat4(1.0f), glm::vec3(sl.position));
				
				glm::vec3 dir = glm::normalize(glm::vec3(sl.direction));
				glm::vec3 up = abs(dir.y) > 0.99f ? glm::vec3(1,0,0) : glm::vec3(0,1,0);
				glm::vec3 right = glm::normalize(glm::cross(up, dir));
				up = glm::cross(dir, right);
				
				glm::mat4 rot(1.0f);
				rot[0] = glm::vec4(right, 0.0f);
				rot[1] = glm::vec4(up, 0.0f);
				rot[2] = glm::vec4(-dir, 0.0f);
				
				float outer_angle = glm::acos(sl.color.w);
				float scale_xy = glm::tan(outer_angle) * 2.0f; 
				
				glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(scale_xy, scale_xy, 1.0f));
				
				cone.model = m * rot * scale;
				cone.albedo = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
				cone.isEmissive = 1.0f;
				
				std::memcpy((char*)model_ubo_mapped + (dynamic_offset), &cone, sizeof(ModelUBO));
				
				vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &descriptor_sets[1], 1, &dynamic_offset);
				vkCmdDraw(fd.command_buffer, cone_line_vertex_count, 1, 0, 0);
				dynamic_offset += aligned_size;
			}
		}

		vkCmdEndRenderPass(fd.command_buffer);
		vkEndCommandBuffer(fd.command_buffer);
	}

}