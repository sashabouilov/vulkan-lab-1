#include "application.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
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
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace application {
	namespace {

	struct Vertex {
		glm::vec3 pos;
		glm::vec3 normal;
		glm::vec2 texCoord;
	};

	struct PointLight { glm::vec4 position; glm::vec4 color; };
	struct SpotLight { glm::vec4 position; glm::vec4 direction; glm::vec4 color; };

	struct SceneUBO {
		glm::mat4 view; glm::mat4 proj; glm::vec4 cameraPos; glm::vec4 ambient;
		glm::vec4 dirLightDir; glm::vec4 dirLightColor; int numPointLights; int numSpotLights; float pad[2];
	};

	struct ModelUBO {
		glm::mat4 model; glm::mat4 normalMat; glm::vec4 albedo; glm::vec4 specColor; float shininess; float isEmissive; float pad[2];
	};

	struct PushConstants {
		float time;
		int use_tricks;
	};

	struct Texture {
		VkImage image;
		VmaAllocation allocation;
		VkImageView view;
		VkSampler sampler;
		VkDescriptorSet descriptorSet;
	};

	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	VkPipeline graphics_pipeline = VK_NULL_HANDLE;
	VkPipeline line_pipeline = VK_NULL_HANDLE;
	VkDescriptorSetLayout descriptor_set_layouts[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
	VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;

	VkBuffer vertex_buffer = VK_NULL_HANDLE; VmaAllocation vertex_buffer_allocation = VK_NULL_HANDLE;
	VkBuffer index_buffer = VK_NULL_HANDLE; VmaAllocation index_buffer_allocation = VK_NULL_HANDLE;
	uint32_t index_count = 0;

	VkBuffer scene_ubo_buffer = VK_NULL_HANDLE; VmaAllocation scene_ubo_allocation = VK_NULL_HANDLE; void* scene_ubo_mapped = nullptr;
	VkBuffer model_ubo_buffer = VK_NULL_HANDLE; VmaAllocation model_ubo_allocation = VK_NULL_HANDLE; void* model_ubo_mapped = nullptr;
	VkDeviceSize model_ubo_aligned_size = 0;

	VkBuffer point_light_buffer = VK_NULL_HANDLE; VmaAllocation point_light_allocation = VK_NULL_HANDLE; void* point_light_mapped = nullptr;
	VkBuffer spot_light_buffer = VK_NULL_HANDLE; VmaAllocation spot_light_allocation = VK_NULL_HANDLE; void* spot_light_mapped = nullptr;

	std::vector<Texture> textures;
	int current_texture_index = 0;

	glm::vec3 camera_pos = {0.0f, 2.0f, 8.0f};
	float camera_pitch = -10.0f; float camera_yaw = -90.0f; float camera_fov = 45.0f;
	SceneUBO scene_ubo_data{};
	std::vector<PointLight> point_lights;
	std::vector<SpotLight> spot_lights;

	glm::vec3 albedo = {0.8f, 0.3f, 0.3f};
	glm::vec3 specular_color = {1.0f, 1.0f, 1.0f};
	float shininess = 32.0f;

	bool use_tricks = false;
	bool show_point_lights = true;
	bool show_spot_lights = true;

	VkBuffer light_cube_buffer = VK_NULL_HANDLE; VmaAllocation light_cube_allocation = VK_NULL_HANDLE; uint32_t light_cube_vertex_count = 0;
	VkBuffer cone_line_buffer = VK_NULL_HANDLE; VmaAllocation cone_line_allocation = VK_NULL_HANDLE; uint32_t cone_line_vertex_count = 0;

	std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);
		if (!file.is_open()) return {};
		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);
		file.seekg(0); file.read(buffer.data(), fileSize); return buffer;
	}

	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
		VkShaderModule shaderModule;
		vkCreateShaderModule(graphics::internal::context.device, &createInfo, nullptr, &shaderModule);
		return shaderModule;
	}

	void createDeviceLocalBuffer(VkBuffer& buffer, VmaAllocation& allocation, VkDeviceSize size, VkBufferUsageFlags usage, const void* data) {
		VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bufferInfo.size = size;
		bufferInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		
		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		vmaCreateBuffer(graphics::internal::context.allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);

		VkBuffer stagingBuffer; VmaAllocation stagingAllocation;
		bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo stagingAllocInfo;
		vmaCreateBuffer(graphics::internal::context.allocator, &bufferInfo, &allocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocInfo);
		std::memcpy(stagingAllocInfo.pMappedData, data, (size_t)size);

		VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		poolInfo.queueFamilyIndex = graphics::internal::context.graphics_queue_index;
		VkCommandPool tempPool; vkCreateCommandPool(graphics::internal::context.device, &poolInfo, nullptr, &tempPool);
		VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		cmdAllocInfo.commandPool = tempPool; cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cmdAllocInfo.commandBufferCount = 1;
		VkCommandBuffer cmd; vkAllocateCommandBuffers(graphics::internal::context.device, &cmdAllocInfo, &cmd);

		VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd, &beginInfo);
		VkBufferCopy copyRegion{}; copyRegion.size = size;
		vkCmdCopyBuffer(cmd, stagingBuffer, buffer, 1, &copyRegion);
		vkEndCommandBuffer(cmd);

		VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
		submitInfo.commandBufferCount = 1; submitInfo.pCommandBuffers = &cmd;
		vkQueueSubmit(graphics::internal::context.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphics::internal::context.graphics_queue);

		vkFreeCommandBuffers(graphics::internal::context.device, tempPool, 1, &cmd);
		vkDestroyCommandPool(graphics::internal::context.device, tempPool, nullptr);
		vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAllocation);
	}

	void createTexture(const char* filename) {
		Texture tex{};
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load(filename, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		if (!pixels) { std::cerr << "Failed to load texture: " << filename << "\n"; return; }
		VkDeviceSize imageSize = texWidth * texHeight * 4;

		VkBuffer stagingBuffer; VmaAllocation stagingAllocation; void* stagingData;
		VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bufferInfo.size = imageSize; bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		VmaAllocationCreateInfo allocInfo{}; allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo allocResult{};
		vmaCreateBuffer(graphics::internal::context.allocator, &bufferInfo, &allocInfo, &stagingBuffer, &stagingAllocation, &allocResult);
		memcpy(allocResult.pMappedData, pixels, imageSize);
		stbi_image_free(pixels);

		VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent = {(uint32_t)texWidth, (uint32_t)texHeight, 1};
		imageInfo.mipLevels = 1; imageInfo.arrayLayers = 1;
		imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT; imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo imgAllocInfo{}; imgAllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		vmaCreateImage(graphics::internal::context.allocator, &imageInfo, &imgAllocInfo, &tex.image, &tex.allocation, nullptr);

		VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		poolInfo.queueFamilyIndex = graphics::internal::context.graphics_queue_index;
		VkCommandPool tempPool; vkCreateCommandPool(graphics::internal::context.device, &poolInfo, nullptr, &tempPool);
		VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		cmdAllocInfo.commandPool = tempPool; cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cmdAllocInfo.commandBufferCount = 1;
		VkCommandBuffer cmd; vkAllocateCommandBuffers(graphics::internal::context.device, &cmdAllocInfo, &cmd);
		VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd, &beginInfo);

		VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
		barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = tex.image; barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		barrier.srcAccessMask = 0; barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		region.imageExtent = {(uint32_t)texWidth, (uint32_t)texHeight, 1};
		vkCmdCopyBufferToImage(cmd, stagingBuffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
		vkEndCommandBuffer(cmd);

		VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submitInfo.commandBufferCount = 1; submitInfo.pCommandBuffers = &cmd;
		vkQueueSubmit(graphics::internal::context.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphics::internal::context.graphics_queue);
		vkFreeCommandBuffers(graphics::internal::context.device, tempPool, 1, &cmd);
		vkDestroyCommandPool(graphics::internal::context.device, tempPool, nullptr);
		vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAllocation);

		VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
		viewInfo.image = tex.image; viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
		viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		vkCreateImageView(graphics::internal::context.device, &viewInfo, nullptr, &tex.view);

		VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
		samplerInfo.magFilter = VK_FILTER_LINEAR; samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT; samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		vkCreateSampler(graphics::internal::context.device, &samplerInfo, nullptr, &tex.sampler);

		textures.push_back(tex);
	}

	void createMappedBuffer(VkBuffer& buffer, VmaAllocation& alloc, void*& mapped, size_t size, VkBufferUsageFlags usage) {
		VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer_info.size = size; buffer_info.usage = usage; buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		VmaAllocationCreateInfo alloc_info{}; alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
		alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo alloc_result{};
		vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &alloc_info, &buffer, &alloc, &alloc_result);
		mapped = alloc_result.pMappedData;
	}

	void createGeometry() {
		std::vector<glm::vec3> base_pos = {
			{ 1.0f, 1.0f/3.0f, 1.0f/3.0f}, { 1.0f/3.0f, 1.0f, 1.0f/3.0f}, { 1.0f/3.0f, 1.0f/3.0f, 1.0f},
			{ 1.0f, -1.0f/3.0f, -1.0f/3.0f}, { 1.0f/3.0f, -1.0f/3.0f, -1.0f}, { 1.0f/3.0f, -1.0f, -1.0f/3.0f},
			{-1.0f/3.0f, 1.0f, -1.0f/3.0f}, {-1.0f/3.0f, 1.0f/3.0f, -1.0f}, {-1.0f, 1.0f/3.0f, -1.0f/3.0f},
			{-1.0f/3.0f, -1.0f/3.0f, 1.0f}, {-1.0f/3.0f, -1.0f, 1.0f/3.0f}, {-1.0f, -1.0f/3.0f, 1.0f/3.0f}
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
			glm::vec3 p0 = base_pos[indices[i]], p1 = base_pos[indices[i+1]], p2 = base_pos[indices[i+2]];
			glm::vec3 normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
			
			auto getUV = [](glm::vec3 p) { return glm::vec2(p.x * 0.5f + 0.5f, p.z * 0.5f + 0.5f); };
			
			vertices.push_back({p0, normal, getUV(p0)});
			vertices.push_back({p1, normal, getUV(p1)});
			vertices.push_back({p2, normal, getUV(p2)});
		}
		index_count = static_cast<uint32_t>(indices.size());
		std::vector<Vertex> seq_vertices;
		std::vector<uint32_t> seq_indices;
		for (size_t i = 0; i < vertices.size(); ++i) {
			seq_vertices.push_back(vertices[i]);
			seq_indices.push_back((uint32_t)i);
		}
		index_count = (uint32_t)seq_indices.size();

		createDeviceLocalBuffer(vertex_buffer, vertex_buffer_allocation, sizeof(Vertex) * seq_vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, seq_vertices.data());
		createDeviceLocalBuffer(index_buffer, index_buffer_allocation, sizeof(uint32_t) * seq_indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, seq_indices.data());
	}

	void createLightCubeBuffer() {
		std::vector<Vertex> vertices = {
			{{ 0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}}, {{ 0.5f,  0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}}, {{ 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
			{{ 0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}}, {{ 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}}, {{ 0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}}, {{-0.5f,  0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}}, {{-0.5f,  0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
			{{-0.5f, -0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}}, {{-0.5f,  0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}}, {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}}, {{ 0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}}, {{ 0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
			{{-0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}}, {{ 0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}, {{-0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}}, {{ 0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}}, {{ 0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
			{{-0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}}, {{ 0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}}, {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}}, {{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}}, {{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
			{{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}}, {{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}}, {{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
			{{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}}, {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}}, {{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
			{{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}}, {{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}}, {{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
		};
		light_cube_vertex_count = 36;
		createDeviceLocalBuffer(light_cube_buffer, light_cube_allocation, sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.data());
	}

	void createConeLineBuffer() {
		std::vector<Vertex> vertices; int segments = 16; float radius = 1.0f, height = 2.0f;
		glm::vec3 tip(0.0f, 0.0f, 0.0f);
		for (int i = 0; i < segments; ++i) {
			float a1 = (float)i / segments * 2.0f * 3.14159f, a2 = (float)(i + 1) / segments * 2.0f * 3.14159f;
			glm::vec3 p1(radius * cos(a1), radius * sin(a1), -height), p2(radius * cos(a2), radius * sin(a2), -height);
			vertices.push_back({tip, {0,0,1}, {0,0}}); vertices.push_back({p1, {0,0,1}, {0,0}});
			vertices.push_back({p1, {0,0,1}, {0,0}}); vertices.push_back({p2, {0,0,1}, {0,0}});
		}
		cone_line_vertex_count = (uint32_t)vertices.size();
		createDeviceLocalBuffer(cone_line_buffer, cone_line_allocation, sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.data());
	}

	void createDescriptorSets() {
		VkDescriptorSetLayoutBinding b0[3]{};
		b0[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		b0[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		b0[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo l0{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; l0.bindingCount = 3; l0.pBindings = b0;
		vkCreateDescriptorSetLayout(graphics::internal::context.device, &l0, nullptr, &descriptor_set_layouts[0]);

		VkDescriptorSetLayoutBinding b1[2]{};
		b1[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		b1[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo l1{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; l1.bindingCount = 2; l1.pBindings = b1;
		vkCreateDescriptorSetLayout(graphics::internal::context.device, &l1, nullptr, &descriptor_set_layouts[1]);

		VkDescriptorPoolSize pool_sizes[] = {
			{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10}
		};
		VkDescriptorPoolCreateInfo p{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		p.poolSizeCount = 4; p.pPoolSizes = pool_sizes; p.maxSets = 10;
		vkCreateDescriptorPool(graphics::internal::context.device, &p, nullptr, &descriptor_pool);

		VkDescriptorSetAllocateInfo a{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		a.descriptorPool = descriptor_pool; a.descriptorSetCount = 1; a.pSetLayouts = &descriptor_set_layouts[0];
		vkAllocateDescriptorSets(graphics::internal::context.device, &a, &scene_descriptor_set);

		VkDescriptorBufferInfo s_info{scene_ubo_buffer, 0, sizeof(SceneUBO)};
		VkDescriptorBufferInfo pl_info{point_light_buffer, 0, VK_WHOLE_SIZE};
		VkDescriptorBufferInfo sl_info{spot_light_buffer, 0, VK_WHOLE_SIZE};
		VkWriteDescriptorSet w0[3]{};
		w0[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &s_info, nullptr};
		w0[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pl_info, nullptr};
		w0[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_descriptor_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sl_info, nullptr};
		vkUpdateDescriptorSets(graphics::internal::context.device, 3, w0, 0, nullptr);

		for (auto& tex : textures) {
			VkDescriptorSetAllocateInfo alloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
			alloc.descriptorPool = descriptor_pool; alloc.descriptorSetCount = 1; alloc.pSetLayouts = &descriptor_set_layouts[1];
			vkAllocateDescriptorSets(graphics::internal::context.device, &alloc, &tex.descriptorSet);

			VkDescriptorBufferInfo m_info{model_ubo_buffer, 0, sizeof(ModelUBO)};
			VkDescriptorImageInfo i_info{tex.sampler, tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

			VkWriteDescriptorSet w[2]{};
			w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tex.descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &m_info, nullptr};
			w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tex.descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &i_info, nullptr, nullptr};
			vkUpdateDescriptorSets(graphics::internal::context.device, 2, w, 0, nullptr);
		}
	}

	void createGraphicsPipeline() {
		auto vert_code = readFile("shaders/shader.vert.spv");
		auto frag_code = readFile("shaders/shader.frag.spv");
		if (vert_code.empty() || frag_code.empty()) return;

		VkShaderModule vert_module = createShaderModule(vert_code);
		VkShaderModule frag_module = createShaderModule(frag_code);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vert_module, "main", nullptr};
		stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, frag_module, "main", nullptr};

		VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
		VkVertexInputAttributeDescription attrs[3] = {
			{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
			{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
			{2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord)}
		};
		VkPipelineVertexInputStateCreateInfo v_in{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
		v_in.vertexBindingDescriptionCount = 1; v_in.pVertexBindingDescriptions = &binding;
		v_in.vertexAttributeDescriptionCount = 3; v_in.pVertexAttributeDescriptions = attrs;

		VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
		ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
		vp.viewportCount = 1; vp.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
		rast.polygonMode = VK_POLYGON_MODE_FILL; rast.lineWidth = 1.0f; rast.cullMode = VK_CULL_MODE_NONE; rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

		VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
		ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
		ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS;

		VkPipelineColorBlendAttachmentState cba{};
		cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
		cb.attachmentCount = 1; cb.pAttachments = &cba;

		VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dyn_s{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
		dyn_s.dynamicStateCount = 2; dyn_s.pDynamicStates = dyn;

		VkPushConstantRange push{};
		push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		push.offset = 0; push.size = sizeof(PushConstants);

		VkPipelineLayoutCreateInfo l_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		l_info.setLayoutCount = 2; l_info.pSetLayouts = descriptor_set_layouts;
		l_info.pushConstantRangeCount = 1; l_info.pPushConstantRanges = &push;
		vkCreatePipelineLayout(graphics::internal::context.device, &l_info, nullptr, &pipeline_layout);

		VkGraphicsPipelineCreateInfo p_info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
		p_info.stageCount = 2; p_info.pStages = stages;
		p_info.pVertexInputState = &v_in; p_info.pInputAssemblyState = &ia;
		p_info.pViewportState = &vp; p_info.pRasterizationState = &rast;
		p_info.pMultisampleState = &ms; p_info.pDepthStencilState = &ds;
		p_info.pColorBlendState = &cb; p_info.pDynamicState = &dyn_s;
		p_info.layout = pipeline_layout; p_info.renderPass = graphics::internal::context.render_pass;

		vkCreateGraphicsPipelines(graphics::internal::context.device, VK_NULL_HANDLE, 1, &p_info, nullptr, &graphics_pipeline);

		ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
		p_info.pInputAssemblyState = &ia;
		vkCreateGraphicsPipelines(graphics::internal::context.device, VK_NULL_HANDLE, 1, &p_info, nullptr, &line_pipeline);

		vkDestroyShaderModule(graphics::internal::context.device, vert_module, nullptr);
		vkDestroyShaderModule(graphics::internal::context.device, frag_module, nullptr);
	}

	} // namespace

	bool initialize() {
		createTexture("textures/tex1.jpg");
		createTexture("textures/tex2.jpg");

		createGeometry();
		createLightCubeBuffer();
		createConeLineBuffer();
		
		createMappedBuffer(scene_ubo_buffer, scene_ubo_allocation, scene_ubo_mapped, sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
		
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &props);
		VkDeviceSize align = props.limits.minUniformBufferOffsetAlignment;
		model_ubo_aligned_size = (sizeof(ModelUBO) + align - 1) & ~(align - 1);
		createMappedBuffer(model_ubo_buffer, model_ubo_allocation, model_ubo_mapped, model_ubo_aligned_size * 100, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
		
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
		vmaDestroyBuffer(context.allocator, index_buffer, index_buffer_allocation);
		vmaDestroyBuffer(context.allocator, light_cube_buffer, light_cube_allocation);
		vmaDestroyBuffer(context.allocator, cone_line_buffer, cone_line_allocation);
		vmaDestroyBuffer(context.allocator, scene_ubo_buffer, scene_ubo_allocation);
		vmaDestroyBuffer(context.allocator, model_ubo_buffer, model_ubo_allocation);
		vmaDestroyBuffer(context.allocator, point_light_buffer, point_light_allocation);
		vmaDestroyBuffer(context.allocator, spot_light_buffer, spot_light_allocation);

		for (auto& tex : textures) {
			vkDestroySampler(context.device, tex.sampler, nullptr);
			vkDestroyImageView(context.device, tex.view, nullptr);
			vmaDestroyImage(context.allocator, tex.image, tex.allocation);
		}
	}

	void update(double time) {
		ImGui::Begin("Scene Controls");
		
		if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::SliderFloat3("Position", glm::value_ptr(camera_pos), -20.0f, 20.0f);
			ImGui::SliderFloat("Yaw", &camera_yaw, -180.0f, 180.0f);
			ImGui::SliderFloat("Pitch", &camera_pitch, -89.0f, 89.0f);
			ImGui::SliderFloat("FOV", &camera_fov, 10.0f, 120.0f);
		}

		if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
			if (ImGui::BeginCombo("Texture", textures.empty() ? "None" : ("Texture " + std::to_string(current_texture_index)).c_str())) {
				for (int i = 0; i < (int)textures.size(); ++i) {
					bool is_selected = (current_texture_index == i);
					if (ImGui::Selectable(("Texture " + std::to_string(i)).c_str(), is_selected)) current_texture_index = i;
					if (is_selected) ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
			ImGui::Checkbox("Nontrivial Sampling", &use_tricks);
			ImGui::ColorEdit3("Albedo Modulate", glm::value_ptr(albedo));
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
				point_lights.push_back({{0, 2, 0, 1.0f}, {1, 1, 1, 1}});
			}
			for (size_t i = 0; i < point_lights.size(); ++i) {
				ImGui::PushID(i);
				ImGui::Text("Light %d", (int)i);
				ImGui::SliderFloat3("Pos", glm::value_ptr(point_lights[i].position), -20.0f, 20.0f);
				ImGui::ColorEdit3("Col", glm::value_ptr(point_lights[i].color));
				ImGui::SliderFloat("Intensity", &point_lights[i].position.w, 0.0f, 50.0f);
				if (ImGui::Button("Remove")) { 
					point_lights.erase(point_lights.begin() + i); 
					ImGui::PopID(); 
					break; 
				}
				ImGui::PopID();
			}
		}

		if (ImGui::CollapsingHeader("Spot Lights")) {
			if (ImGui::Button("Add Spot Light") && spot_lights.size() < 100) {
				spot_lights.push_back({{0, 5, 0, 1.0f}, {0, -1, 0, glm::cos(glm::radians(15.0f))}, {1, 1, 1, glm::cos(glm::radians(25.0f))}});
			}
			for (size_t i = 0; i < spot_lights.size(); ++i) {
				ImGui::PushID(i + 100);
				ImGui::Text("Spot %d", (int)i);
				ImGui::SliderFloat3("Pos", glm::value_ptr(spot_lights[i].position), -20.0f, 20.0f);
				ImGui::SliderFloat3("Dir", glm::value_ptr(spot_lights[i].direction), -1.0f, 1.0f);
				
				ImGui::SliderFloat("Intensity", &spot_lights[i].position.w, 0.0f, 100.0f); 
				
				float inner = glm::degrees(glm::acos(spot_lights[i].direction.w));
				float outer = glm::degrees(glm::acos(spot_lights[i].color.w));
				
				// Защита от того, чтобы Inner не стал больше Outer
				bool changed = false;
				if (ImGui::SliderFloat("Inner Angle", &inner, 1.0f, 44.0f)) {
					if (inner >= outer) outer = inner + 1.0f;
					changed = true;
				}
				if (ImGui::SliderFloat("Outer Angle", &outer, 2.0f, 45.0f)) {
					if (outer <= inner) inner = outer - 1.0f;
					changed = true;
				}
				if (changed) {
					spot_lights[i].direction.w = glm::cos(glm::radians(inner));
					spot_lights[i].color.w = glm::cos(glm::radians(outer));
				}
					
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

		scene_ubo_data.view = glm::lookAt(camera_pos, camera_pos + front, glm::vec3(0, 1, 0));
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

		VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
		rp.renderPass = graphics::internal::context.render_pass; rp.framebuffer = fd.framebuffer;
		rp.renderArea = { {0, 0}, graphics::internal::context.swapchain_extent };
		rp.clearValueCount = 2; rp.pClearValues = clear_values;

		vkCmdBeginRenderPass(fd.command_buffer, &rp, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

		VkViewport viewport{0.0f, 0.0f, (float)graphics::internal::context.swapchain_extent.width, (float)graphics::internal::context.swapchain_extent.height, 0.0f, 1.0f};
		vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
		VkRect2D scissor{{0, 0}, graphics::internal::context.swapchain_extent};
		vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &scene_descriptor_set, 0, nullptr);

		std::vector<ModelUBO> model_ubos;
		
		ModelUBO tetra{};
		tetra.model = glm::mat4(1.0f);
		tetra.normalMat = glm::transpose(glm::inverse(tetra.model));
		tetra.albedo = glm::vec4(albedo, 1.0f);
		tetra.specColor = glm::vec4(specular_color, 1.0f);
		tetra.shininess = shininess;
		tetra.isEmissive = 0.0f;
		model_ubos.push_back(tetra);

		for (const auto& pl : point_lights) {
			ModelUBO l{};
			l.model = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(pl.position)), glm::vec3(0.2f));
			l.normalMat = glm::transpose(glm::inverse(l.model));
			l.albedo = pl.color; l.isEmissive = 1.0f; model_ubos.push_back(l);
		}
		for (const auto& sl : spot_lights) {
			ModelUBO l{};
			l.model = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(sl.position)), glm::vec3(0.2f));
			l.normalMat = glm::transpose(glm::inverse(l.model));
			l.albedo = sl.color; l.isEmissive = 1.0f; model_ubos.push_back(l);
		}

		for (size_t i = 0; i < model_ubos.size(); ++i) {
			std::memcpy((char*)model_ubo_mapped + (i * model_ubo_aligned_size), &model_ubos[i], sizeof(ModelUBO));
		}

		uint32_t dynamic_offset = 0;
		VkDeviceSize offsets[] = {0};
		PushConstants pc{ (float)glfwGetTime(), use_tricks ? 1 : 0 };

		if (!textures.empty()) {
			vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &textures[current_texture_index].descriptorSet, 1, &dynamic_offset);
		}
		vkCmdPushConstants(fd.command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, offsets);
		vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
		dynamic_offset += model_ubo_aligned_size;

		pc.use_tricks = 0;
		vkCmdPushConstants(fd.command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
		
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &light_cube_buffer, offsets);
		for (size_t i = 0; i < model_ubos.size() - 1; ++i) {
			if (!textures.empty()) {
				vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &textures[0].descriptorSet, 1, &dynamic_offset);
			}
			vkCmdDraw(fd.command_buffer, light_cube_vertex_count, 1, 0, 0);
			dynamic_offset += model_ubo_aligned_size;
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
				glm::mat4 rot(1.0f); rot[0] = glm::vec4(right, 0.0f); rot[1] = glm::vec4(up, 0.0f); rot[2] = glm::vec4(-dir, 0.0f);
				float scale_xy = glm::tan(glm::acos(sl.color.w)) * 2.0f; 
				cone.model = m * rot * glm::scale(glm::mat4(1.0f), glm::vec3(scale_xy, scale_xy, 1.0f));
				cone.albedo = glm::vec4(1.0f); cone.isEmissive = 1.0f;
				std::memcpy((char*)model_ubo_mapped + dynamic_offset, &cone, sizeof(ModelUBO));
				
				vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &textures[0].descriptorSet, 1, &dynamic_offset);
				vkCmdDraw(fd.command_buffer, cone_line_vertex_count, 1, 0, 0);
				dynamic_offset += model_ubo_aligned_size;
			}
		}

		vkCmdEndRenderPass(fd.command_buffer);
		vkEndCommandBuffer(fd.command_buffer);
	}

} // namespace application