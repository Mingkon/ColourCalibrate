# ColourCalibrate

This repository shows how to calibrate colour with standard colour checker.



It is implemented with python3 and cpp

In cpp it is accelerated with CUDA



#### 0.  requirements

   for py  both opencv and  lib-opencv are needed
   for cpp both opencv and opencv-contrib are need, and make sure their versions are match.

#### 1. algorithm

$$
stdColourBlock =   imgColourBlock * Coefficient^{T}
$$



$$
stdColourBlock = \left(
\begin{matrix}
s_{11} &s_{12}  &s_{13} \\
s_{21} &s_{22}  &s_{23} \\
\cdots &\cdots &\cdots \\
s_{241} &s_{242} &s_{243} \\
\end{matrix}
\right)_{24 * 3}
$$


$$
imgColourBlock = \left(
\begin{matrix}
i_{11} &i_{12}  &i_{13} \\
i_{21} &i_{22}  &i_{23} \\
\cdots &\cdots &\cdots \\
i_{241} &i_{242}  &i_{243} \\
\end{matrix}
\right)_{24 * 3}
$$

$$
Coefficient^{T} = \left(
\begin{matrix}
c_{11} &c_{12}  &c_{1n} \\
c_{21} &c_{22}  &c_{2n} \\
c_{31} &c_{32}  &c_{3n} \\
\end{matrix}
\right)_{3 * 3}
$$



stdColourBlock  -- standard colour space

imgColourBlock -- pictured colour space

Coefficient^{T}   -- colour calibrate matrix



with linear algebra:
$$
Coefficient^{T} = (imgColourBlock * imgColourBlock^{T})^{-1} *(imgColourBlock * stdColourBlock^{T})
$$
$$
C = (I * I^{T})^{-1} * (I * S^{T})
$$

we get  colour calibrate matrix.


- (for each pixel):

$$
calibratedImg = Coefficient^{T} * rawImg
$$


$$
calibratedImg = 
\left(
\begin{matrix}
s_{11} \\
s_{21} \\
s_{31} \\
\end{matrix}
\right)_{3 * 1}
$$

$$
Coefficient^{T} = \left(
\begin{matrix}
c_{11} &c_{12}  &c_{13} \\
c_{21} &c_{22}  &c_{23} \\
c_{31} &c_{32}  &c_{33} \\
\end{matrix}\right)_{3 * 3}
$$

$$
rawImg = 
\left(
\begin{matrix}
i_{11} \\
i_{21} \\
i_{31} \\
\end{matrix}
\right)_{3* 1}
$$







colour space can be [R,G,B],[L,A,B],[H,S,V] and so on.

if we choose stdColourBlock  and imgColourBlock with shape: 3 * 24, we have 



$$
S = C^{T} * I
$$

$$
\left(
\begin{matrix}
s_{11} &s_{12} &\cdots &s_{124} \\
s_{21} &s_{22} &\cdots &s_{224} \\
s_{31} &s_{32} &\cdots &s_{324} \\
\end{matrix}
\right)_{3 * 24} 

=

\left(
\begin{matrix}
c_{11} &c_{12}  &c_{13} \\
c_{21} &c_{22}  &c_{23} \\
c_{31} &c_{32}  &c_{33} \\
\end{matrix}
\right)_{3 * 3} 

*

\left(
\begin{matrix}
i_{11} &i_{12} &\cdots &i_{124} \\
i_{21} &i_{22} &\cdots &i_{224} \\
i_{31} &i_{32} &\cdots &i_{324} \\
\end{matrix}
\right)_{3 * 24}
$$



then
$$
calibratedImg = C^{T} * rawImg
$$


#### 2 discuss

In the example, the dark part of calibrated image is dotted with noise, and  more details need to be done. 
