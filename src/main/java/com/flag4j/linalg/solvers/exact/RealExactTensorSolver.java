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

package com.flag4j.linalg.solvers.exact;


import com.flag4j.core.Shape;
import com.flag4j.dense.Matrix;
import com.flag4j.dense.Tensor;
import com.flag4j.dense.Vector;

/**
 * Solver for solving a real well determined linear tensor equation {@code A*X=B} in an exact sense.
 */
public class RealExactTensorSolver extends ExactTensorSolver<Tensor, Matrix, Vector> {


    /**
     * Creates an exact tensor solver for solving a well determined linear tensor equation {@code A*X=B}
     * for {@code X} in an exact sense.
     */
    public RealExactTensorSolver() {
        super(new RealExactSolver());
    }


    /**
     * Initializes matrix for equivalent linear matrix equation.
     *
     * @param A    Tensor to convert to matrix.
     * @param prod Product of all axis lengths in {@code A}.
     * @return A matrix with the same entries as tensor {@code A} with shape (prod, prod).
     */
    @Override
    protected Matrix initMatrix(Tensor A, int prod) {
        return new Matrix(prod, prod, A.entries);
    }


    /**
     * Initializes vector for equivalent linear matrix equation.
     *
     * @param B Tensor to convert to vector.
     * @return Flattens tensor {@code B} and converts to a vector.
     */
    @Override
    protected Vector initVector(Tensor B) {
        return new Vector(B.entries);
    }


    /**
     * Wraps solution as a tensor and reshapes to the proper shape.
     *
     * @param x           Vector solution to matrix linear equation which is equivalent to the tensor equation {@code A*X=B}.
     * @param outputShape Shape for the solution tensor {@code X}.
     * @return The solution {@code X} to the linear tensor equation {@code A*X=B}.
     */
    @Override
    protected Tensor wrap(Vector x, Shape outputShape) {
        return new Tensor(outputShape, x.entries);
    }
}
