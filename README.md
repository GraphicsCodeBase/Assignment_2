# Seam Carving - Assignment 2

Quick setup guide for building and running the project using vcpkg and Visual Studio (MSVC).

---

## Prerequisites

- Visual Studio 2019/2022 with C++ Desktop Development workload
- Git for Windows
- CMake (usually included with Visual Studio)

---

## Setup Instructions

### Step 1: Install vcpkg

Open Command Prompt or PowerShell and run:

```bash
cd C:\
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
```

### Step 2: Install OpenCV

```bash
vcpkg install opencv:x64-windows
```

Note: First-time installation takes 15-30 minutes (OpenCV compiles from source).

### Step 3: Integrate vcpkg

```bash
vcpkg integrate install
```

You should see: `CMake projects should use: "-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake"`

---

## Building the Project

### Step 4: Open Project in Visual Studio

1. Launch Visual Studio
2. Click "Open a local folder"
3. Navigate to and select the `Assignment_2` folder
4. Visual Studio will automatically detect `CMakeLists.txt`

### Step 5: Configure CMake

1. Visual Studio should auto-configure CMake
2. If prompted for toolchain file, use:
   ```
   C:/vcpkg/scripts/buildsystems/vcpkg.cmake
   ```

Alternative - Manual Configuration:
- Open CMake Settings (Project → CMake Settings)
- Add to CMake command arguments:
  ```
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
  ```

### Step 6: Build

1. Select "Release" configuration (top toolbar)
2. Click Build → Build All (or press `Ctrl+Shift+B`)
3. Wait for build to complete (~2-5 minutes)

---

## Running the Programs

### Step 7: Prepare Test Image

Create folder structure:
```
Assignment_2/
├── Images/
│   └── tower.jpg    ← Place your test image here
└── out/
    └── build/
```

### Step 8: Run from Visual Studio

Option A - Run directly:
1. Right-click `main.cpp` in Solution Explorer
2. Click "Set as Startup Item"
3. Press `F5` or click "Run"

Option B - Run from executable:
1. Navigate to: `Assignment_2\out\build\x64-Release\`
2. Double-click `SeamCarving.exe`

---

## Available Programs

### 1. Main Seam Carving Tool (`main.cpp`)
Interactive menu with DP, Graph-Cuts, and visualization features.

Key Features:
- `[1]` Custom resize
- `[5]` Visualization mode (Bonus)
- `[10]` Graph-Cuts (Question 5c)
- `[11]` DP vs Graph-Cuts comparison

### 2. DP vs Greedy Comparison (`CompareDPGreed.cpp`)
Benchmarks both algorithms and generates detailed statistics (Question 4).

Outputs:
- Performance metrics
- Quality comparison
- `benchmark_results.txt`
- Comparison images

### 3. Greedy Tool (`GreedyAlgo.cpp`)
Standalone greedy algorithm implementation.

---

## Quick Test

After building, test with these steps:

1. Run `SeamCarving.exe`
2. Choose option `[1]` (Custom size)
3. Enter width: `700`
4. Enter height: `500`
5. Wait for processing (~30-60 seconds)
6. View the result comparison

---

## Troubleshooting

### "Could not load image" error
Fix: Ensure `Images/tower.jpg` exists relative to the executable location.

### "OpenCV not found" during build
Fix:
```bash
cd C:\vcpkg
vcpkg integrate install
```
Then rebuild in Visual Studio.

### Build is slow
Fix: Use parallel builds:
- Visual Studio → Tools → Options → Projects and Solutions → Build and Run
- Set "maximum number of parallel project builds" to `4` or higher

---

## Project Structure

```
Assignment_2/
├── main.cpp                   # Main program
├── CompareDPGreed.cpp         # Question 4
├── GreedyAlgo.cpp             # Greedy standalone
├── seam_carving_bonus.h/cpp   # Bonus features
├── GridGraph_2D_4C.h          # Graph-Cuts (Q5c)
├── Images/
│   └── tower.jpg              # Test image
└── CMakeLists.txt
```

---

## Performance Notes

| Algorithm | Speed | Quality |
|-----------|-------|---------|
| DP | 36.5s | Best (13,602 avg energy) |
| Greedy | 20.3s | Good (26,568 avg energy) |
| Graph-Cuts | 107s | Best (global optimum) |

---

## Team 18 - Algorithm Analysis Assignment 2

---

That's it! You should now be able to build and run all three programs.
