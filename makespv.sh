#!/bin/bash

/home/qxy/Downloads/1.4.321.1/x86_64/bin/glslangValidator --target-env vulkan1.0 /home/qxy/Downloads/VulkanTutorial/code/32_cube.frag -o frag_cube.spv
/home/qxy/Downloads/1.4.321.1/x86_64/bin/glslangValidator --target-env vulkan1.0 /home/qxy/Downloads/VulkanTutorial/code/32_cube.vert -o vert_cube.spv
/home/qxy/Downloads/1.4.321.1/x86_64/bin/glslangValidator --target-env vulkan1.0 /home/qxy/Downloads/VulkanTutorial/code/32_light.frag -o frag_light.spv
/home/qxy/Downloads/1.4.321.1/x86_64/bin/glslangValidator --target-env vulkan1.0 /home/qxy/Downloads/VulkanTutorial/code/32_light.vert -o vert_light.spv
