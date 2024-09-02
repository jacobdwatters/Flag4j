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

package org.flag4j.linalg.solvers.exact;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CTensorOld;
import org.flag4j.arrays_old.dense.CVectorOld;


/**
 * Solver for solving a complex well determined linear tensor equation {@code A*X=B} in an exact sense.
 */
public class ComplexExactTensorSolver extends ExactTensorSolver<CTensorOld, CMatrixOld, CVectorOld> {


    /**
     * Creates an exact tensor solver for solving a well determined linear tensor equation {@code A*X=B}
     * for {@code X} in an exact sense.
     */
    public ComplexExactTensorSolver() {
        super(new ComplexExactSolver());
    }


    /**
     * Initializes matrix for equivalent linear matrix equation.
     *
     * @param A    TensorOld to convert to matrix.
     * @param prod Product of all axis lengths in {@code A}.
     * @return A matrix with the same entries as tensor {@code A} with shape (prod, prod).
     */
    @Override
    protected CMatrixOld initMatrix(CTensorOld A, int prod) {
        return new CMatrixOld(prod, prod, A.entries);
    }


    /**
     * Initializes vector for equivalent linear matrix equation.
     *
     * @param B TensorOld to convert to vector.
     * @return Flattens tensor {@code B} and converts to a vector.
     */
    @Override
    protected CVectorOld initVector(CTensorOld B) {
        return new CVectorOld(B.entries);
    }


    /**
     * Wraps solution as a tensor and reshapes to the proper shape.
     *
     * @param x           VectorOld solution to matrix linear equation which is equivalent to the tensor equation {@code A*X=B}.
     * @param outputShape Shape for the solution tensor {@code X}.
     * @return The solution {@code X} to the linear tensor equation {@code A*X=B}.
     */
    @Override
    protected CTensorOld wrap(CVectorOld x, Shape outputShape) {
        return new CTensorOld(outputShape, x.entries);
    }
}
