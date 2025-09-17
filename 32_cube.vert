#version 450

#define CUBE_ROW  20
#define CUBE_COL  20
#define CUBE_DEEP 20
/*
struct StorageBufferObject {
    vec3 pos[CUBE_ROW*CUBE_COL];
    vec3 thetap[CUBE_ROW*CUBE_COL];
    mat4 model_pos[CUBE_ROW*CUBE_COL];
    mat4 model[CUBE_ROW*CUBE_COL];
};
*/
layout(std140,binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(std140, binding = 2) buffer StorageBufferObject {
    mat4 model_pos[CUBE_DEEP*CUBE_ROW*CUBE_COL];
    mat4 rmodel[CUBE_DEEP*CUBE_ROW*CUBE_COL];
    mat4 model[CUBE_DEEP*CUBE_ROW*CUBE_COL];
    vec3 thetap[CUBE_DEEP*CUBE_ROW*CUBE_COL];
    vec3 pos[CUBE_DEEP*CUBE_ROW*CUBE_COL];
} ssbo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    gl_Position = ubo.proj * ubo.view * ssbo.model[gl_InstanceIndex] * vec4(inPosition, 1.0);
    //gl_Position = ubo.proj * ubo.view * vec4(inPosition, 1.0);
    //gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = inColor; 
    fragTexCoord = inTexCoord;
}
