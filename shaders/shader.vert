#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec3 fragColor;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view; mat4 proj; vec4 cameraPos; vec4 ambient;
    vec4 dirLightDir; vec4 dirLightColor; int numPointLights; int numSpotLights;
} scene;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model; mat4 normalMat; vec4 albedo; vec4 specColor; float shininess; float isEmissive;
} model;

layout(push_constant) uniform PushConstants {
    float time;
    int use_tricks;
} pc;

void main() {
    vec4 world_pos = model.model * vec4(inPosition, 1.0);
    gl_Position = scene.proj * scene.view * world_pos;
    fragPos = world_pos.xyz;
    fragNormal = mat3(model.normalMat) * inNormal;
    
    if (pc.use_tricks > 0) {
        fragTexCoord = inTexCoord + vec2(sin(pc.time), cos(pc.time)) * 0.1;
    } else {
        fragTexCoord = inTexCoord;
    }
    
    fragColor = (inPosition * 0.5 + 0.5);
}