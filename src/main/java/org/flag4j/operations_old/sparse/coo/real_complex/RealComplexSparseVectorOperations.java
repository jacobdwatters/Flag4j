/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.operations_old.sparse.coo.real_complex;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.sparse.CooCVector;
import org.flag4j.arrays_old.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * This class contains low level implementations of operations_old on a real sparse tensor and a complex sparse tensor.
 */
public final class RealComplexSparseVectorOperations {

    private RealComplexSparseVectorOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Adds a real number to each entry of a sparse vector, including the zero entries.
     * @param src Sparse vector to add value to.
     * @param a Value to add to the {@code src} sparse vector.
     * @return The result of adding the specified value to the sparse vector.
     */
    public static CVectorOld add(CooVector src, CNumber a) {
        CNumber[] dest = new CNumber[src.size];
        Arrays.fill(dest, a);

        for(int i=0; i<src.entries.length; i++) {
            int idx = src.indices[i];
            dest[idx] = dest[idx].add(src.entries[i]);
        }

        return new CVectorOld(dest);
    }


    /**
     * Adds a real number to each entry of a sparse vector, including the zero entries.
     * @param src Sparse vector to add value to.
     * @param a Value to add to the {@code src} sparse vector.
     * @return The result of adding the specified value to the sparse vector.
     */
    public static CVectorOld add(CooCVector src, double a) {
        CNumber[] dest = new CNumber[src.size];
        ArrayUtils.fill(dest, a);

        for(int i=0; i<src.entries.length; i++) {
            int idx = src.indices[i];
            dest[idx] = dest[idx].add(src.entries[i]);
        }

        return new CVectorOld(dest);
    }


    /**
     * Subtracts a real number from each entry of a sparse vector, including the zero entries.
     * @param src Sparse vector to subtract value from.
     * @param a Value to subtract from the {@code src} sparse vector.
     * @return The result of subtracting the specified value from the sparse vector.
     */
    public static CVectorOld sub(CooVector src, CNumber a) {
        CNumber[] dest = new CNumber[src.size];
        Arrays.fill(dest, a.addInv());

        for(int i=0; i<src.entries.length; i++) {
            int idx = src.indices[i];
            dest[idx] = dest[idx].add(src.entries[i]);
        }

        return new CVectorOld(dest);
    }


    /**
     * Computes the element-wise vector addition between a real sparse vector and a complex sparse vector.
     * Both sparse vectors are assumed to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the addition. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the addition. Indices assumed to be sorted lexicographically.
     * @return The result of the vector addition.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static CooCVector add(CooCVector src1, CooVector src2) {
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

        return new CooCVector(
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
    public static CooCVector sub(CooCVector src1, CooVector src2) {
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

        return new CooCVector(
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
    public static CooCVector sub(CooVector src1, CooCVector src2) {
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

        return new CooCVector(
                src1.size,
                values.toArray(new CNumber[0]),
                indices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Computes the element-wise vector multiplication between two real sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the multiplication. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the multiplication. Indices assumed to be sorted lexicographically.
     * @return The result of the vector multiplication.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static CooCVector elemMult(CooCVector src1, CooVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        List<CNumber> values = new ArrayList<>(src1.entries.length);
        List<Integer> indices = new ArrayList<>(src1.entries.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.entries.length && src2Counter < src2.entries.length) {
            if(src1.indices[src1Counter]==src2.indices[src2Counter]) {
                // Then indices match, add product of elements.
                values.add(src1.entries[src1Counter].mult(src2.entries[src2Counter]));
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                src1Counter++;
            } else {
                src2Counter++;
            }
        }

        return new CooCVector(
                src1.size,
                values.toArray(CNumber[]::new),
                indices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Computes the inner product of a real and complex sparse vector. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the inner product. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the inner product. Indices assumed to be sorted lexicographically.
     * @return The result of the vector inner product.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static CNumber inner(CooCVector src1, CooVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        CNumber product = CNumber.ZERO;

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.entries.length && src2Counter < src2.entries.length) {
            if(src1.indices[src1Counter]==src2.indices[src2Counter]) {
                // Then indices match, add product of elements.
                product = product.add(src1.entries[src1Counter].mult(src2.entries[src2Counter]));
            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                src1Counter++;
            } else {
                src2Counter++;
            }
        }

        return product;
    }


    /**
     * Computes the inner product of a real and complex sparse vector. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the inner product. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the inner product. Indices assumed to be sorted lexicographically.
     * @return The result of the vector inner product.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static CNumber inner(CooVector src1, CooCVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        CNumber product = CNumber.ZERO;

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.entries.length && src2Counter < src2.entries.length) {
            if(src1.indices[src1Counter]==src2.indices[src2Counter]) {
                // Then indices match, add product of elements.
                product = product.add(src2.entries[src2Counter].conj().mult(src1.entries[src1Counter]));
                src1Counter++;
                src2Counter++;
            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                src1Counter++;
            } else {
                src2Counter++;
            }
        }

        return product;
    }


    /**
     * Computes the vector outer product between a complex sparse vector and a real sparse vector.
     * @param src1 Entries of the first sparse vector in the outer product.
     * @param src2 Second sparse vector in the outer product.
     * @return The matrix resulting from the vector outer product.
     */
    public static CMatrixOld outerProduct(CooCVector src1, CooVector src2) {
        CNumber[] dest = new CNumber[src2.size*src1.size];
        Arrays.fill(dest, CNumber.ZERO);

        int destRow;
        int index1;
        int index2;

        for(int i=0; i<src1.entries.length; i++) {
            index1 = src1.indices[i];
            destRow = index1*src1.size;

            for(int j=0; j<src2.entries.length; j++) {
                index2 = src2.indices[j];

                dest[destRow + index2] = src1.entries[i].mult(src2.entries[j]);
            }
        }

        return new CMatrixOld(src1.size, src2.size, dest);
    }


    /**
     * Computes the vector outer product between a complex sparse vector and a real sparse vector.
     * @param src1 Entries of the first sparse vector in the outer product.
     * @param src2 Second sparse vector in the outer product.
     * @return The matrix resulting from the vector outer product.
     */
    public static CMatrixOld outerProduct(CooVector src1, CooCVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CNumber[] dest = new CNumber[src2.size*src1.size];
        Arrays.fill(dest, CNumber.ZERO);

        int destRow;
        int index1;
        int index2;

        for(int i=0; i<src1.entries.length; i++) {
            index1 = src1.indices[i];
            destRow = index1*src1.size;

            for(int j=0; j<src2.entries.length; j++) {
                index2 = src2.indices[j];

                dest[destRow + index2] = src2.entries[j].mult(src1.entries[i]);
            }
        }

        return new CMatrixOld(src1.size, src2.size, dest);
    }
}
