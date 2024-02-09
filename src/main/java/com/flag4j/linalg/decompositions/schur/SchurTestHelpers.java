/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package com.flag4j.linalg.decompositions.schur;

import com.flag4j.core.MatrixMixin;
import com.flag4j.dense.CMatrix;
import com.flag4j.dense.Matrix;

import static com.flag4j.util.Flag4jConstants.EPS_F64;

class SchurTestHelpers {

    static void printBulge(Matrix src) {
        System.out.print("Bulge structure:\n[[  ");

        for(int i=0; i<src.numRows; i++) {
            if(i>0) System.out.print(" [  ");

            for(int j=0; j<src.numRows; j++) {
                if(i <= j+1) {
                    if( i==j+1 && Math.abs(src.get(i, j)) < EPS_F64) System.out.print("   ");
                    else System.out.print("X  ");
                } else {
                    if(Math.abs(src.get(i, j)) < EPS_F64) {
                        System.out.print("   ");
                    } else {
                        System.out.print("+  ");
                    }
                }
            }

            System.out.print("]");
            if(i < src.numRows-1) {
                System.out.println();
            }
        }

        System.out.println("]\n");
    }


    public static void printAsJavaArray(Object... args) {
        for(Object arg : args) {
            if(arg instanceof MatrixMixin) {
                printAsJavaArray((MatrixMixin<?, ?, ?, ?, ?, ?, ?>) arg);
            } else {
                System.out.print(arg.toString());
            }
        }
    }

    private static <T extends MatrixMixin<?, ?, ?, ?, ?, ?, ?>> void printAsJavaArray(T A) {
        System.out.println("{");

        for(int i=0; i<A.numRows(); i++) {
            System.out.print("\t{");
            for(int j=0; j<A.numCols(); j++) {
                if(A instanceof CMatrix) {
                    CMatrix B = (CMatrix) A;
                    System.out.print("new CNumber(" + B.get(i, j).re + ", " + B.get(i, j).im + ")");
                } else {
                    // Then must be real.
                    System.out.print(A.get(i, j));
                }

                if(j < A.numCols()-1) {
                    System.out.print(", ");
                }
            }
            System.out.print("}");

            if(i < A.numRows()-1) {
                System.out.println(",");
            }
        }

        System.out.println("\n};");
    }
}
