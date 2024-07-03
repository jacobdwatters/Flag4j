/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package org.flag4j.core;

/**
 * This interface specifies comparisons which all tensors (i.e. matrices and vectors) should implement.
 */
public interface TensorComparisonsMixin {


    /**
     * Checks if this tensor only contains zeros.
     * @return True if this tensor only contains zeros. Otherwise, returns false.
     */
    boolean isZeros();


    /**
     * Checks if this tensor only contains ones.
     * @return True if this tensor only contains ones. Otherwise, returns false.
     */
    boolean isOnes();


    /**
     * Checks if this tensor is equal to a specified Object. Note, this method differs from {@link #tensorEquals(TensorBase)} as, in
     * this method, the types of the objects must match where in {@link #tensorEquals(TensorBase)} the typed need not match.
     * @param B Object to compare this tensor to.
     * @return True if B is the same object type as this tensor, has the same shape, and each element of the two tensors are
     * numerically equal. Otherwise, returns false.
     */
    @Override
    boolean equals(Object B);


    /**
     * <p>Checks if two tensors are equal. Note, this method is much more permissive than {@link #equals(Object)} as it allows for
     * comparisons with any tensor, matrix, or vector. If the shapes are equal and entries are element-wise equivalent then this
     * method returns true regardless of the types of the two tensors.</p>
     *
     * <p>xFor Example, a {@link org.flag4j.dense.Tensor Dense Tensor} of rank 2 with the same values as a</p>
     *
     * {@link org.flag4j.sparse.CsrMatrix Sparse CSR Matrix} will be considered equal.
     * @param B Tensor, matrix, or vector to compare to this tensor.
     * @return True if both tensors (or matrix/vector) have the same shape and all entries are numerically equivalent by index. This
     * accounts for possible zero values in sparse objects. Returns false if the tensors do not have the same shape or if the
     * tensors differ in value at <i>any</i>  index.
     */
    boolean tensorEquals(TensorBase<?, ?, ?, ?, ?, ?, ?> B);
}
