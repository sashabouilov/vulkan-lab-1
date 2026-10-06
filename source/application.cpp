#include "application.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <fstream>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>

namespace application {

//  Внутреннее состояние и вспомогательные функции

namespace {

struct Vertex {
	glm::vec3 pos;
};

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec4 color;
};

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline graphics_pipeline = VK_NULL_HANDLE;
VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_buffer_allocation = VK_NULL_HANDLE;
VkBuffer index_buffer = VK_NULL_HANDLE;
VmaAllocation index_buffer_allocation = VK_NULL_HANDLE;
uint32_t index_count = 0;

VkBuffer uniform_buffer = VK_NULL_HANDLE;
VmaAllocation uniform_buffer_allocation = VK_NULL_HANDLE;
void* uniform_buffer_mapped = nullptr;

bool use_perspective = true;
glm::vec3 position = {0.0f, 0.0f, 0.0f};
glm::vec3 rotation = {0.0f, 0.0f, 0.0f};
glm::vec3 scale    = {1.0f, 1.0f, 1.0f};
float color[3]     = {0.8f, 0.3f, 0.3f};

bool animate = false;
float animation_speed = 1.0f;
float animation_radius = 2.0f;
float animation_angle = 0.0f;

glm::vec3 camera_pos   = {0.0f, 0.0f, 5.0f};
glm::vec3 camera_front = {0.0f, 0.0f, -1.0f};
glm::vec3 camera_up    = {0.0f, 1.0f, 0.0f};

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

void createVertexAndIndexBuffers() {
	// Вершины усечённого тетраэдра (12 штук)
	std::vector<Vertex> vertices = {
		{{ 1.0f,      1.0f/3.0f,  1.0f/3.0f}},
		{{ 1.0f/3.0f, 1.0f,      1.0f/3.0f}},
		{{ 1.0f/3.0f, 1.0f/3.0f, 1.0f     }},
		{{ 1.0f,     -1.0f/3.0f, -1.0f/3.0f}},
		{{ 1.0f/3.0f,-1.0f/3.0f, -1.0f    }},
		{{ 1.0f/3.0f,-1.0f,     -1.0f/3.0f}},
		{{-1.0f/3.0f, 1.0f,     -1.0f/3.0f}},
		{{-1.0f/3.0f, 1.0f/3.0f,-1.0f     }},
		{{-1.0f,      1.0f/3.0f,-1.0f/3.0f}},
		{{-1.0f/3.0f,-1.0f/3.0f, 1.0f     }},
		{{-1.0f/3.0f,-1.0f,      1.0f/3.0f}},
		{{-1.0f,     -1.0f/3.0f, 1.0f/3.0f}},
	};

	// 4 треугольника + 4 шестиугольника (разбитые на треугольники)
	std::vector<uint32_t> indices = {
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
		9, 10, 11,
		0, 3, 4,  0, 4, 7,  0, 7, 6,  0, 6, 1,
		0, 3, 5,  0, 5, 10, 0, 10, 9, 0, 9, 2,
		1, 6, 8,  1, 8, 11, 1, 11, 9, 1, 9, 2,
		4, 7, 8,  4, 8, 11, 4, 11, 10, 4, 10, 5,
	};
	index_count = static_cast<uint32_t>(indices.size());

	VkDeviceSize vertex_size = sizeof(Vertex) * vertices.size();
	VkDeviceSize index_size  = sizeof(uint32_t) * indices.size();

	VkBufferCreateInfo buffer_info{};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
	alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
					   VMA_ALLOCATION_CREATE_MAPPED_BIT;

	VmaAllocationInfo alloc_result{};

	buffer_info.size = vertex_size;
	buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &alloc_info,
					&vertex_buffer, &vertex_buffer_allocation, &alloc_result);
	std::memcpy(alloc_result.pMappedData, vertices.data(), vertex_size);

	buffer_info.size = index_size;
	buffer_info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &alloc_info,
					&index_buffer, &index_buffer_allocation, &alloc_result);
	std::memcpy(alloc_result.pMappedData, indices.data(), index_size);
}

void createUniformBuffer() {
	VkBufferCreateInfo buffer_info{};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size = sizeof(UniformBufferObject);
	buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
	alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
					   VMA_ALLOCATION_CREATE_MAPPED_BIT;

	VmaAllocationInfo alloc_result{};
	vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &alloc_info,
					&uniform_buffer, &uniform_buffer_allocation, &alloc_result);
	uniform_buffer_mapped = alloc_result.pMappedData;
}

void createDescriptorSetLayout() {
	VkDescriptorSetLayoutBinding ubo_binding{};
	ubo_binding.binding = 0;
	ubo_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	ubo_binding.descriptorCount = 1;
	ubo_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutCreateInfo layout_info{};
	layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layout_info.bindingCount = 1;
	layout_info.pBindings = &ubo_binding;

	vkCreateDescriptorSetLayout(graphics::internal::context.device, &layout_info,
								nullptr, &descriptor_set_layout);
}

void createDescriptorPool() {
	VkDescriptorPoolSize pool_size{};
	pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	pool_size.descriptorCount = 1;

	VkDescriptorPoolCreateInfo pool_info{};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes = &pool_size;
	pool_info.maxSets = 1;

	vkCreateDescriptorPool(graphics::internal::context.device, &pool_info,
						   nullptr, &descriptor_pool);
}

void createDescriptorSet() {
	VkDescriptorSetAllocateInfo alloc_info{};
	alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	alloc_info.descriptorPool = descriptor_pool;
	alloc_info.descriptorSetCount = 1;
	alloc_info.pSetLayouts = &descriptor_set_layout;

	vkAllocateDescriptorSets(graphics::internal::context.device, &alloc_info, &descriptor_set);

	VkDescriptorBufferInfo buffer_info{};
	buffer_info.buffer = uniform_buffer;
	buffer_info.offset = 0;
	buffer_info.range = sizeof(UniformBufferObject);

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = descriptor_set;
	write.dstBinding = 0;
	write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	write.descriptorCount = 1;
	write.pBufferInfo = &buffer_info;

	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

void createGraphicsPipeline() {
	auto vert_code = readFile("shaders/shader.vert.spv");
	auto frag_code = readFile("shaders/shader.frag.spv");
	if (vert_code.empty() || frag_code.empty()) {
		std::cerr << "Failed to load shaders. Check 'shaders/' folder for .spv files.\n";
		return;
	}

	VkShaderModule vert_module = createShaderModule(vert_code);
	VkShaderModule frag_module = createShaderModule(frag_code);

	VkPipelineShaderStageCreateInfo stages[2]{};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vert_module;
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = frag_module;
	stages[1].pName = "main";

	VkVertexInputBindingDescription binding{};
	binding.binding = 0;
	binding.stride = sizeof(Vertex);
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attr{};
	attr.binding = 0;
	attr.location = 0;
	attr.format = VK_FORMAT_R32G32B32_SFLOAT;
	attr.offset = offsetof(Vertex, pos);

	VkPipelineVertexInputStateCreateInfo vertex_input{};
	vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertex_input.vertexBindingDescriptionCount = 1;
	vertex_input.pVertexBindingDescriptions = &binding;
	vertex_input.vertexAttributeDescriptionCount = 1;
	vertex_input.pVertexAttributeDescriptions = &attr;

	VkPipelineInputAssemblyStateCreateInfo input_assembly{};
	input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	VkPipelineViewportStateCreateInfo viewport_state{};
	viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport_state.viewportCount = 1;
	viewport_state.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_NONE;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineDepthStencilStateCreateInfo depth_stencil{};
	depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth_stencil.depthTestEnable = VK_TRUE;
	depth_stencil.depthWriteEnable = VK_TRUE;
	depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;

	VkPipelineColorBlendAttachmentState blend_attachment{};
	blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
									  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	blend_attachment.blendEnable = VK_FALSE;

	VkPipelineColorBlendStateCreateInfo color_blending{};
	color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blending.attachmentCount = 1;
	color_blending.pAttachments = &blend_attachment;

	VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dynamic_state{};
	dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic_state.dynamicStateCount = 2;
	dynamic_state.pDynamicStates = dynamic_states;

	VkPipelineLayoutCreateInfo layout_info{};
	layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts = &descriptor_set_layout;

	vkCreatePipelineLayout(graphics::internal::context.device, &layout_info,
						   nullptr, &pipeline_layout);

	VkGraphicsPipelineCreateInfo pipeline_info{};
	pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeline_info.stageCount = 2;
	pipeline_info.pStages = stages;
	pipeline_info.pVertexInputState = &vertex_input;
	pipeline_info.pInputAssemblyState = &input_assembly;
	pipeline_info.pViewportState = &viewport_state;
	pipeline_info.pRasterizationState = &rasterizer;
	pipeline_info.pMultisampleState = &multisampling;
	pipeline_info.pDepthStencilState = &depth_stencil;
	pipeline_info.pColorBlendState = &color_blending;
	pipeline_info.pDynamicState = &dynamic_state;
	pipeline_info.layout = pipeline_layout;
	pipeline_info.renderPass = graphics::internal::context.render_pass;
	pipeline_info.subpass = 0;

	vkCreateGraphicsPipelines(graphics::internal::context.device, VK_NULL_HANDLE,
							  1, &pipeline_info, nullptr, &graphics_pipeline);

	vkDestroyShaderModule(graphics::internal::context.device, vert_module, nullptr);
	vkDestroyShaderModule(graphics::internal::context.device, frag_module, nullptr);
}

void updateUniformBuffer() {
	UniformBufferObject ubo{};

	glm::mat4 model = glm::mat4(1.0f);
	if (animate) {
		float x = animation_radius * std::cos(animation_angle);
		float z = animation_radius * std::sin(animation_angle);
		model = glm::translate(model, glm::vec3(x, 0.0f, z));
	} else {
		model = glm::translate(model, position);
	}
	model = glm::rotate(model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
	model = glm::rotate(model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
	model = glm::scale(model, scale);
	ubo.model = model;

	ubo.view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);

	float aspect = static_cast<float>(graphics::internal::context.swapchain_extent.width) /
				   static_cast<float>(graphics::internal::context.swapchain_extent.height);
	if (use_perspective) {
		ubo.proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
	} else {
		float s = 4.0f;
		ubo.proj = glm::ortho(-s * aspect, s * aspect, -s, s, 0.1f, 100.0f);
	}
	ubo.proj[1][1] *= -1.0f;

	ubo.color = glm::vec4(color[0], color[1], color[2], 1.0f);

	std::memcpy(uniform_buffer_mapped, &ubo, sizeof(ubo));
}

} // namespace

//  Публичные функции

bool initialize() {
	createVertexAndIndexBuffers();
	createUniformBuffer();
	createDescriptorSetLayout();
	createDescriptorPool();
	createDescriptorSet();
	createGraphicsPipeline();
	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	if (graphics_pipeline != VK_NULL_HANDLE)
		vkDestroyPipeline(context.device, graphics_pipeline, nullptr);
	if (pipeline_layout != VK_NULL_HANDLE)
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	if (descriptor_pool != VK_NULL_HANDLE)
		vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	if (descriptor_set_layout != VK_NULL_HANDLE)
		vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);
	if (uniform_buffer != VK_NULL_HANDLE)
		vmaDestroyBuffer(context.allocator, uniform_buffer, uniform_buffer_allocation);
	if (index_buffer != VK_NULL_HANDLE)
		vmaDestroyBuffer(context.allocator, index_buffer, index_buffer_allocation);
	if (vertex_buffer != VK_NULL_HANDLE)
		vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_buffer_allocation);
}

void update(double time) {
	static double last_time = 0.0;
	double delta = time - last_time;
	last_time = time;

	if (animate) {
		animation_angle += animation_speed * static_cast<float>(delta);
		if (animation_angle > 2.0f * 3.14159265f) {
			animation_angle -= 2.0f * 3.14159265f;
		}
	}

	ImGui::Begin("Lab 1 - Truncated Tetrahedron");

	ImGui::Text("Projection");
	ImGui::RadioButton("Perspective", reinterpret_cast<int*>(&use_perspective), 1);
	ImGui::SameLine();
	ImGui::RadioButton("Orthographic", reinterpret_cast<int*>(&use_perspective), 0);

	ImGui::Separator();
	ImGui::Text("Transform");
	ImGui::SliderFloat3("Position", glm::value_ptr(position), -5.0f, 5.0f);
	ImGui::SliderFloat3("Rotation", glm::value_ptr(rotation), -180.0f, 180.0f);
	ImGui::SliderFloat3("Scale", glm::value_ptr(scale), 0.1f, 3.0f);

	ImGui::Separator();
	ImGui::Text("Animation");
	ImGui::Checkbox("Animate (play/pause)", &animate);
	ImGui::SliderFloat("Speed", &animation_speed, 0.0f, 5.0f);
	ImGui::SliderFloat("Radius", &animation_radius, 0.0f, 5.0f);

	ImGui::Separator();
	ImGui::Text("Color");
	ImGui::ColorEdit3("Shape Color", color);

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	updateUniformBuffer();

	vkResetCommandBuffer(fd.command_buffer, 0);

	VkCommandBufferBeginInfo begin_info{};
	begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(fd.command_buffer, &begin_info);

	VkClearValue clear_values[2];
	clear_values[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}};
	clear_values[1].depthStencil = {1.0f, 0};

	VkRenderPassBeginInfo render_pass_info{};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	render_pass_info.renderPass = graphics::internal::context.render_pass;
	render_pass_info.framebuffer = fd.framebuffer;
	render_pass_info.renderArea.offset = {0, 0};
	render_pass_info.renderArea.extent = graphics::internal::context.swapchain_extent;
	render_pass_info.clearValueCount = 2;
	render_pass_info.pClearValues = clear_values;

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = static_cast<float>(graphics::internal::context.swapchain_extent.width);
	viewport.height = static_cast<float>(graphics::internal::context.swapchain_extent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.offset = {0, 0};
	scissor.extent = graphics::internal::context.swapchain_extent;
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
							pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

	VkBuffer vertex_buffers[] = { vertex_buffer };
	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vertex_buffers, offsets);
	vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);

	vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application