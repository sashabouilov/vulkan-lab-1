#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(cross(dFdx(fragPos), dFdy(fragPos)));

    vec3 light_dir = normalize(vec3(0.5, 1.0, 0.8));
    float diff = abs(dot(normal, light_dir));
    float ambient = 0.3;
    float lighting = ambient + (1.0 - ambient) * diff;

    outColor = vec4(fragColor * lighting, 1.0);
}