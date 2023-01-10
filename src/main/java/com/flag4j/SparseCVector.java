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

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.SparseVectorBase;
import com.flag4j.core.VectorOrientation;
import com.flag4j.util.ArrayUtils;

/**
 * Complex sparse vector.
 */
public class SparseCVector extends SparseVectorBase<CNumber[]> {


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     */
    public SparseCVector(int size) {
        super(size, 0, VectorOrientation.COL, new CNumber[0], new int[0]);
    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseCVector(int size, int[] nonZeroEntries, int[] indices) {
        super(size, nonZeroEntries.length, VectorOrientation.COL, new CNumber[nonZeroEntries.length], indices);
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseCVector(int size, double[] nonZeroEntries, int[] indices) {
        super(size, nonZeroEntries.length, VectorOrientation.COL, new CNumber[nonZeroEntries.length], indices);
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseCVector(int size, CNumber[] nonZeroEntries, int[] indices) {
        super(size, nonZeroEntries.length, VectorOrientation.COL, nonZeroEntries, indices);
    }


//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseCVector(int[] entries) {
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
//        super.entries = new CNumber[super.nonZeroEntries()];
//        for(int i=0; i<super.entries.length; i++) {
//            super.entries[i] = new CNumber(nonZeroEntries.get(i));
//        }
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
//    public SparseCVector(double[] entries) {
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
//        super.entries = new CNumber[super.nonZeroEntries()];
//        for(int i=0; i<super.entries.length; i++) {
//            super.entries[i] = new CNumber(nonZeroEntries.get(i));
//        }
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
//    public SparseCVector(CNumber[] entries) {
//        super(entries.length, VectorOrientations.COL);
//
//        ArrayList<CNumber> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i].re!=0 && entries[i].im!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = new CNumber[super.nonZeroEntries()];
//        for(int i=0; i<super.entries.length; i++) {
//            super.entries[i] = nonZeroEntries.get(i).copy();
//        }
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param orientation Orientation of the vector.
     */
    public SparseCVector(int size, VectorOrientation orientation) {
        super(size, 0, orientation, new CNumber[0], new int[0]);
    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @param orientation Orientation of the vector.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseCVector(int size, int[] nonZeroEntries, int[] indices, VectorOrientation orientation) {
        super(size, nonZeroEntries.length, orientation, new CNumber[nonZeroEntries.length], indices);
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @param orientation Orientation of the vector.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseCVector(int size, double[] nonZeroEntries, int[] indices, VectorOrientation orientation) {
        super(size, nonZeroEntries.length, orientation, new CNumber[nonZeroEntries.length], indices);
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @param orientation Orientation of the vector.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseCVector(int size, CNumber[] nonZeroEntries, int[] indices, VectorOrientation orientation) {
        super(size, nonZeroEntries.length, orientation, nonZeroEntries, indices);
    }


    // TODO: These methods (for all sparse tensor classes) will be moved into factory methods
//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @param orientation Orientation of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseCVector(int[] entries, VectorOrientations orientation) {
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
//        super.entries = new CNumber[super.nonZeroEntries()];
//        for(int i=0; i<super.entries.length; i++) {
//            super.entries[i] = new CNumber(nonZeroEntries.get(i));
//        }
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
//    public SparseCVector(double[] entries, VectorOrientations orientation) {
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
//        super.entries = new CNumber[super.nonZeroEntries()];
//        for(int i=0; i<super.entries.length; i++) {
//            super.entries[i] = new CNumber(nonZeroEntries.get(i));
//        }
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
//    public SparseCVector(CNumber[] entries, VectorOrientations orientation) {
//        super(entries.length, orientation);
//
//        ArrayList<CNumber> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i].re!=0 && entries[i].im!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = new CNumber[super.nonZeroEntries()];
//        for(int i=0; i<super.entries.length; i++) {
//            super.entries[i] = nonZeroEntries.get(i).copy();
//        }
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }


    /**
     * Constructs a complex sparse vector whose size, orientation, non-zero entries, and indices are specified
     * by another complex sparse vector.
     * @param a Vector to copy.
     */
    public SparseCVector(SparseCVector a) {
        super(a.size(), a.nonZeroEntries(), a.getOrientation(), new CNumber[a.nonZeroEntries()], a.indices.clone());
        ArrayUtils.copy2CNumber(a.entries, super.entries);
    }
}
