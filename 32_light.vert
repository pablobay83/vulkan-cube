#version 450

#define LIGHTER_NUM 512

layout(std140,binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(std140,binding = 1) uniform UniformStorageBufferObject {
    mat4 model[LIGHTER_NUM];
} usbo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    gl_Position = ubo.proj * ubo.view * usbo.model[gl_InstanceIndex] * vec4(inPosition, 1.0);
    //gl_Position = ubo.proj * ubo.view * vec4(inPosition, 1.0);
    //gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = inColor; 
    fragTexCoord = inTexCoord;
}
