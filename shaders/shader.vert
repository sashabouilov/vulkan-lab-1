#version 450

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 color;
} ubo;

void main() {
    vec4 world_pos = ubo.model * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * world_pos;
    fragPos = world_pos.xyz;
    fragColor = (inPosition * 0.5 + 0.5) * ubo.color.rgb;
}