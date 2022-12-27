/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.flag4j;


import com.flag4j.util.RandomTensor;

public class Testing {

    public static double[] withSqrt(double[] arr) {
        double[] sqrt = new double[arr.length];

        for(int i=0; i<sqrt.length; i++) {
            sqrt[i] = Math.sqrt(arr[i]);
        }

        return sqrt;
    }

    public static double[] withPow(double[] arr) {
        double[] sqrt = new double[arr.length];

        for(int i=0; i<sqrt.length; i++) {
            sqrt[i] = Math.pow(arr[i], 1.0/2.0);
        }

        return sqrt;
    }


    public static void main(String[] args) {

        RandomTensor rng = new RandomTensor(42l);
        double[] A = rng.getRandomMatrix(10000, 10000).entries;

        long startTime = System.nanoTime();
        withSqrt(A);
        long endTime = System.nanoTime();
        System.out.println("withSqrt: " + (endTime-startTime)*1.0e-6 + " ms");

        startTime = System.nanoTime();
        withPow(A);
        endTime = System.nanoTime();
        System.out.println("withPow: " + (endTime-startTime)*1.0e-6 + " ms");
    }
}
