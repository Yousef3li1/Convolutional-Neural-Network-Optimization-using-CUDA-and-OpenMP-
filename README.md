# 🚀 Convolutional Neural Network Optimization using CUDA and OpenMP  
 

---

## 📖 **Project Overview**  

This project focuses on optimizing **Convolutional Neural Networks (CNNs)** by leveraging parallel computing techniques using **CUDA** and **OpenMP**. CNNs are widely used in computer vision tasks, but due to their computational intensity, they can suffer from slow execution times. To address this, the project explores and implements parallel execution models using **GPU acceleration** (CUDA) and **multithreading** (OpenMP) to achieve significant speedup and enhance performance.  


---

## ⚙️ **Technologies Used**  

- **CUDA:** For parallel execution on NVIDIA GPUs.  
- **OpenMP:** For parallel execution using CPU multithreading.  
- **Jupyter Notebook:** For testing and experimentation.  
- **C/C++:** As the core programming languages.  

---

## ✅ **Key Features**  

1. **Parallel Execution:**  
   - CUDA kernels for GPU-based parallel processing of CNN forward passes.  
   - OpenMP directives to parallelize nested loops in CNN layers.  

2. **Optimized CNN Layers:**  
   - Convolutional layers  
   - Pooling layers  
   - ReLU activation layers  

3. **Memory Management:**  
   Efficient memory access and management using CUDA device pointers and OpenMP shared memory optimizations.  

4. **Performance Measurement:**  
   - Measure speedup compared to sequential execution.  
   - Record accuracy and efficiency metrics.

---

## 🔍 **CUDA Code Implementation**  

### 🚀 **Introduction**  
The CUDA implementation focuses on optimizing the **forward pass** of a CNN, particularly in the **convolutional layers**, by distributing the computational workload across GPU threads.

### 🛠️ **Methodology**  
1. **Parallel Execution:**  
   CUDA kernels distribute the input data among GPU threads using the syntax `<numBlocks, threadsPerBlock>`.  
2. **Memory Management:**  
   Device pointers are used to access and modify data efficiently in the GPU memory.  

### 📌 **CUDA Functions**  
- **Kernel Function (doGPU):** Handles the convolutional computations in parallel.  
- **conv_forward_cu Function:** Configures and launches the CUDA kernel, ensuring synchronization between threads.  

---

## 🔍 **OpenMP Code Implementation**  

### 🚀 **Introduction**  
The OpenMP implementation optimizes the CNN by parallelizing critical sections of the code, particularly nested loops, using **OpenMP directives**.

### 🛠️ **Methodology**  
1. **Layer Optimization:**  
   The convolutional and pooling layers are parallelized using the `#pragma omp parallel for` directive.  
2. **Resource Management:**  
   OpenMP efficiently distributes the workload across available CPU cores for optimal resource usage.  

### 📌 **Optimized Functions**  
- **conv_forward:** Parallelized using OpenMP to execute nested loops concurrently.  
- **pool_forward:** Optimized to reduce redundant computations and increase throughput.  

---

## 📊 **Performance Comparison**  

| **Metric**        | **CUDA (GPU)**                 | **OpenMP (CPU)**                 |
|------------------|--------------------------------|----------------------------------|
| **Technology**    | GPU parallelization using CUDA kernels | CPU multithreading using OpenMP directives |
| **Focus**         | Convolutional layer optimization | Convolutional and pooling layers optimization |
| **Speedup**       | Higher (due to large-scale parallelism) | Moderate (based on available CPU cores) |

---

## 📉 **Benchmark Results**  

| **Metric**         | **Benchmark** | **Optimized** | **Speedup**   |
|-------------------|---------------|---------------|---------------|
| **Accuracy**       | 78.25%        | 78.25%        | -             |
| **Execution Time** | 22,224,240 ms | 2,593,726 ms  | 11.99x        |

---

