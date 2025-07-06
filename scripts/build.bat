@echo off
echo [CUDA] Building vector_add...

nvcc -o vector_add src\vector_add.cu

if %errorlevel% neq 0 (
    echo Build failed.
    exit /b %errorlevel%
) else (
    echo Build successful! Run: vector_add
)
