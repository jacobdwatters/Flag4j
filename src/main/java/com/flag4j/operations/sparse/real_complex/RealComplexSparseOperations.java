/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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

package com.flag4j.operations.sparse.real_complex;

import com.flag4j.CVector;
import com.flag4j.SparseCVector;
import com.flag4j.SparseVector;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * This class contains low level implementations of operations on a real sparse tensor and a complex sparse tensor.
 */
public class RealComplexSparseOperations {

    private RealComplexSparseOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Adds a real number to each entry of a sparse vector, including the zero entries.
     * @param src Sparse vector to add value to.
     * @param a Value to add to the {@code src} sparse vector.
     * @return The result of adding the specified value to the sparse vector.
     */
    public static CVector add(SparseVector src, CNumber a) {
        CNumber[] dest = new CNumber[src.size];
        ArrayUtils.fill(dest, a);

        for(int i=0; i<src.entries.length; i++) {
            dest[src.indices[i]].addEq(src.entries[i]);
        }

        return new CVector(dest);
    }


    /**
     * Adds a real number to each entry of a sparse vector, including the zero entries.
     * @param src Sparse vector to add value to.
     * @param a Value to add to the {@code src} sparse vector.
     * @return The result of adding the specified value to the sparse vector.
     */
    public static CVector add(SparseCVector src, double a) {
        CNumber[] dest = new CNumber[src.size];
        ArrayUtils.fill(dest, a);

        for(int i=0; i<src.entries.length; i++) {
            dest[src.indices[i]].addEq(src.entries[i]);
        }

        return new CVector(dest);
    }


    /**
     * Subtracts a real number from each entry of a sparse vector, including the zero entries.
     * @param src Sparse vector to subtract value from.
     * @param a Value to subtract from the {@code src} sparse vector.
     * @return The result of subtracting the specified value from the sparse vector.
     */
    public static CVector sub(SparseVector src, CNumber a) {
        CNumber[] dest = new CNumber[src.size];
        ArrayUtils.fill(dest, a.addInv());

        for(int i=0; i<src.entries.length; i++) {
            dest[src.indices[i]].addEq(src.entries[i]);
        }

        return new CVector(dest);
    }


    /**
     * Computes the element-wise vector addition between a real sparse vector and a complex sparse vector.
     * Both sparse vectors are assumed to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the addition. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the addition. Indices assumed to be sorted lexicographically.
     * @return The result of the vector addition.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static SparseCVector add(SparseCVector src1, SparseVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        List<CNumber> values = new ArrayList<>(src1.entries.length);
        List<Integer> indices = new ArrayList<>(src1.entries.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.entries.length && src2Counter < src2.entries.length) {
            if(src1.indices[src1Counter] == src2.indices[src2Counter]) {
                values.add(src1.entries[src1Counter].add(src2.entries[src2Counter]));
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                values.add(src1.entries[src1Counter]);
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
            } else {
                values.add(new CNumber(src2.entries[src2Counter]));
                indices.add(src2.indices[src2Counter]);
                src2Counter++;
            }
        }

        // Finish inserting the rest of the values.
        if(src1Counter < src1.entries.length) {
            for(int i=src1Counter; i<src1.entries.length; i++) {
                values.add(src1.entries[i]);
                indices.add(src1.indices[i]);
            }
        } else if(src2Counter < src2.entries.length) {
            for(int i=src2Counter; i<src2.entries.length; i++) {
                values.add(new CNumber(src2.entries[i]));
                indices.add(src2.indices[i]);
            }
        }

        return new SparseCVector(
                src1.size,
                values.toArray(new CNumber[0]),
                indices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Computes the element-wise vector subtraction between a real sparse vector and a complex sparse vector.
     * Both sparse vectors are assumed to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the subtraction. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the subtraction. Indices assumed to be sorted lexicographically.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static SparseCVector sub(SparseCVector src1, SparseVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        List<CNumber> values = new ArrayList<>(src1.entries.length);
        List<Integer> indices = new ArrayList<>(src1.entries.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.entries.length && src2Counter < src2.entries.length) {
            if(src1.indices[src1Counter] == src2.indices[src2Counter]) {
                values.add(src1.entries[src1Counter].sub(src2.entries[src2Counter]));
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                values.add(src1.entries[src1Counter]);
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
            } else {
                values.add(new CNumber(-src2.entries[src2Counter]));
                indices.add(src2.indices[src2Counter]);
                src2Counter++;
            }
        }

        // Finish inserting the rest of the values.
        if(src1Counter < src1.entries.length) {
            for(int i=src1Counter; i<src1.entries.length; i++) {
                values.add(src1.entries[i]);
                indices.add(src1.indices[i]);
            }
        } else if(src2Counter < src2.entries.length) {
            for(int i=src2Counter; i<src2.entries.length; i++) {
                values.add(new CNumber(-src2.entries[i]));
                indices.add(src2.indices[i]);
            }
        }

        return new SparseCVector(
                src1.size,
                values.toArray(new CNumber[0]),
                indices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Computes the element-wise vector subtraction between a real sparse vector and a complex sparse vector.
     * Both sparse vectors are assumed to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the subtraction. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the subtraction. Indices assumed to be sorted lexicographically.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static SparseCVector sub(SparseVector src1, SparseCVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        List<CNumber> values = new ArrayList<>(src1.entries.length);
        List<Integer> indices = new ArrayList<>(src1.entries.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.entries.length && src2Counter < src2.entries.length) {
            if(src1.indices[src1Counter] == src2.indices[src2Counter]) {
                values.add(new CNumber(src1.entries[src1Counter]).sub(src2.entries[src2Counter]));
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                values.add(new CNumber(src1.entries[src1Counter]));
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
            } else {
                values.add(src2.entries[src2Counter].addInv());
                indices.add(src2.indices[src2Counter]);
                src2Counter++;
            }
        }

        // Finish inserting the rest of the values.
        if(src1Counter < src1.entries.length) {
            for(int i=src1Counter; i<src1.entries.length; i++) {
                values.add(new CNumber(src1.entries[i]));
                indices.add(src1.indices[i]);
            }
        } else if(src2Counter < src2.entries.length) {
            for(int i=src2Counter; i<src2.entries.length; i++) {
                values.add(src2.entries[i].addInv());
                indices.add(src2.indices[i]);
            }
        }

        return new SparseCVector(
                src1.size,
                values.toArray(new CNumber[0]),
                indices.stream().mapToInt(Integer::intValue).toArray()
        );
    }
}
