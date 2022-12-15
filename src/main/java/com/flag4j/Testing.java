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

import com.flag4j.operations.RealDenseTranspose;
import com.flag4j.util.RandomTensor;

import java.util.Arrays;

public class Testing {

    static void checkEq(double[] arr1, double[] arr2) {
        if(arr1.length != arr2.length) {
            throw new RuntimeException("Arrays are not equal lengths.");
        }

        for(int i=0; i<arr1.length; i++) {
            if(arr1[i] != arr2[i]) {
                throw new RuntimeException("Arrays are not equal: " + arr1[i] + ", " + arr2[i]);
            }
        }
    }


    public static void main(String[] args) {
        int numRows = 10000;
        int numCols = 30000;

        long startTime, endTime;

        RandomTensor rng = new RandomTensor();
        double[] A = rng.getRandomMatrix(numRows, numCols).entries;

        // Transpose algorithm 2
        startTime = System.nanoTime();
        RealDenseTranspose.standardMatrix(A, numRows, numCols);
        endTime = System.nanoTime();
        System.out.println("Standard: " + (endTime-startTime)/1000000.0 + " ms");

        // Transpose algorithm 3
        startTime = System.nanoTime();
        RealDenseTranspose.blockedMatrix(A, numRows, numCols);
        endTime = System.nanoTime();
        System.out.println("Blocked: " + (endTime-startTime)/1000000.0 + " ms");

        // Transpose algorithm 4
        startTime = System.nanoTime();
        RealDenseTranspose.standardMatrixConcurrent(A, numRows, numCols);
        endTime = System.nanoTime();
        System.out.println("Standard Concurrent: " + (endTime-startTime)/1000000.0 + " ms");

        // Transpose algorithm 5
        startTime = System.nanoTime();
        RealDenseTranspose.blockedMatrixConcurrent(A, numRows, numCols);
        endTime = System.nanoTime();
        System.out.println("Blocked Concurrent: " + (endTime-startTime)/1000000.0 + " ms");
    }
}
