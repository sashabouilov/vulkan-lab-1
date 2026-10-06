#pragma once

// Указываем ImGui, что невалидный ID текстуры — это -1, а не 0.
// Это решает проблему с assert в версиях 1.92+.
#define ImTextureID_Invalid ((ImTextureID)-1)