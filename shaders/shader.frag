#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 position; // w = intensity
    vec4 color;    // rgb = color
};

struct SpotLight {
    vec4 position;  // w = intensity
    vec4 direction; // w = cos(inner)
    vec4 color;     // w = cos(outer)
};

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambient;
    vec4 dirLightDir;
    vec4 dirLightColor;
    int numPointLights;
    int numSpotLights;
} scene;

layout(set = 0, binding = 1) readonly buffer PointLights { PointLight lights[]; } pointLights;
layout(set = 0, binding = 2) readonly buffer SpotLights { SpotLight lights[]; } spotLights;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    mat4 normalMat;
    vec4 albedo;
    vec4 specColor;
    float shininess;
    float isEmissive;
} model;

layout(set = 1, binding = 1) uniform sampler2D albedoTexture;

layout(push_constant) uniform PushConstants {
    float time;
    int use_tricks;
} pc;

vec3 blinn_phong(vec3 N, vec3 V, vec3 L, vec3 lightColor, vec3 baseAlbedo) {
    float diff = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), model.shininess);
    
    return (baseAlbedo * diff + model.specColor.rgb * spec) * lightColor;
}

void main() {
    if (model.isEmissive > 0.5) {
        outColor = model.albedo;
        return;
    }

    vec4 texColor = texture(albedoTexture, fragTexCoord);
    
    vec3 baseAlbedo = texColor.rgb * model.albedo.rgb;

    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene.cameraPos.xyz - fragPos);
    
    if (dot(N, V) < 0.0) {
        N = -N;
    }
    
    vec3 result = scene.ambient.rgb * baseAlbedo;

    vec3 L_dir = normalize(-scene.dirLightDir.xyz);
    result += blinn_phong(N, V, L_dir, scene.dirLightColor.rgb, baseAlbedo);

    for (int i = 0; i < scene.numPointLights; ++i) {
        vec3 light_vec = pointLights.lights[i].position.xyz - fragPos;
        float dist = length(light_vec);
        vec3 L = light_vec / dist;
        
        float attenuation = pointLights.lights[i].position.w / (dist * dist);
        result += blinn_phong(N, V, L, pointLights.lights[i].color.rgb * attenuation, baseAlbedo);
    }

    for (int i = 0; i < scene.numSpotLights; ++i) {
        vec3 light_vec = spotLights.lights[i].position.xyz - fragPos;
        float dist = length(light_vec);
        vec3 L = light_vec / dist;
        
        float theta = dot(L, normalize(-spotLights.lights[i].direction.xyz));
        float inner = spotLights.lights[i].direction.w;
        float outer = spotLights.lights[i].color.w;
        float epsilon = inner - outer;
        float intensity = clamp((theta - outer) / epsilon, 0.0, 1.0);
        
        float attenuation = (spotLights.lights[i].position.w / (dist * dist)) * intensity;
        result += blinn_phong(N, V, L, spotLights.lights[i].color.rgb * attenuation, baseAlbedo);
    }

    outColor = vec4(result, 1.0);
}