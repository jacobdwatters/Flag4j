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

import com.flag4j.core.SparseVectorBase;

import java.util.Arrays;

/**
 * Real sparse vector.
 */
public class SparseVector extends SparseVectorBase<double[]> {


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     */
    public SparseVector(int size) {
        super(size, 0, new double[0], new int[0]);
    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseVector(int size, int[] nonZeroEntries, int[] indices) {
        super(size, nonZeroEntries.length,
                Arrays.stream(nonZeroEntries).asDoubleStream().toArray(),
                indices);
    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseVector(int size, double[] nonZeroEntries, int[] indices) {
        super(size, nonZeroEntries.length, nonZeroEntries, indices);
    }


//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseVector(int[] entries) {
//        super(entries.length, VectorOrientations.COL);
//
//        ArrayList<Integer> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = nonZeroEntries.stream().mapToDouble(Integer::doubleValue).toArray();
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }
//
//
//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseVector(double[] entries) {
//        super(entries.length, VectorOrientations.COL);
//
//        ArrayList<Double> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = nonZeroEntries.stream().mapToDouble(Double::doubleValue).toArray();
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }


//
//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @param orientation Orientation of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseVector(int[] entries, VectorOrientations orientation) {
//        super(entries.length, orientation);
//
//        ArrayList<Integer> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = nonZeroEntries.stream().mapToDouble(Integer::doubleValue).toArray();
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }
//
//
//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @param orientation Orientation of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseVector(double[] entries, VectorOrientations orientation) {
//        super(entries.length, orientation);
//
//        ArrayList<Double> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = nonZeroEntries.stream().mapToDouble(Double::doubleValue).toArray();
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }


    /**
     * Constructs a sparse vector whose non-zero values, indices, and size are specified by another sparse vector.
     * @param a Sparse vector to copy
     */
    public SparseVector(SparseVector a) {
        super(a.size(), a.nonZeroEntries(), a.entries.clone(), a.indices.clone());
    }
}
