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

import org.flag4j.core.*;
import org.flag4j.linalg.solvers.LinearSolver;
import org.flag4j.linalg.solvers.LinearTensorSolver;
import org.flag4j.util.ParameterChecks;


/**
 * <p>Solves a well determined system of equations {@code A*X=B} in an exact sense where A, X, and B are tensors.</p>
 * <p>If the system is not well determined, i.e. {@code A} is 'square' and full rank, then use a
 * least-squares solver.</p>
 *
 * @param <T> Type of tensor in equation to solve.
 * @param <U> Matrix type equivalent of tensor to solve.
 * @param <V> Vector type equivalent of tensor to solve.
 */
public abstract class ExactTensorSolver<
        T extends TensorBase<T, ?, ?, ?, ?, ?, ?>,
        U extends MatrixMixin<U, ?, ?, ?, ?, V, ?>,
        V extends VectorMixin<V, ?, ?, ?, ?, U, ?, ?>>
        implements LinearTensorSolver<T> {

    /**
     * Solver to solve a linear matrix equation {@code C*X=d} for {@code X} where {@code C} and {@code X} are matrices and
     * {@code d} is a vector.
     */
    private final LinearSolver<U, V> matrixSolver;


    /**
     * Creates an exact tensor solver which will use reform the problem as a matrix linear system and use the provided
     * solver to solve the tensor system.
     * @param matrixSolver Matrix solver to use as the solver for the equivalent matrix system.
     */
    protected ExactTensorSolver(LinearSolver<U, V> matrixSolver) {
        this.matrixSolver = matrixSolver;
    }


    /**
     * Solves the linear tensor equation given by {@code A*X=B} for the tensor {@code X}. All indices of {@code X} are summed over in
     * the tensor product with the rightmost indices of {@code A} as if by
     * {@link TensorExclusiveMixin#tensorDot(TensorBase, int[], int[])  A.tensorDot(X, M, N)} where
     * {@code M = new int[]{X.rank()-1, X.rank(), X.rank()+1, ..., A.rank()-1}} and {@code N = new int[]{0, 1, ..., X.rank()-1}}
     * @param A Coefficient tensor in the linear system.
     * @param B Tensor of constants in the linear system.
     * @return The solution to {@code x} in the linear system {@code A*X=B}.
     */
    @Override
    public T solve(T A, T B) {
        int rank = A.getRank();
        Shape outputShape = getOutputShape(A, B, rank); // Compute output shape.

        int prod = 1;
        for(int k : outputShape.dims) {
            prod *= k;
        }

        // Ensure that prod(a.shape.dims[j]) == prod(a.shape.dims[k]) for all b.rank()<=j<a.rank() and 0<=k<b.rank()
        checkSize(A.totalEntries().intValueExact(), prod);

        // Reform the problem as a
        U aMat = initMatrix(A, prod); // Reshape and convert tensor A to matrix.
        V bVec = initVector(B); // Flatten tensor B to a vector.

        // Solve equivalent matrix equation and reshape to be the solution tensor.
        return wrap(matrixSolver.solve(aMat, bVec), outputShape);
    }


    /**
     * Constructs the shape of the output.
     * @param A Tensor corresponding to {@code A} in {@code A*X=B}.
     * @param B Tensor corresponding to {@code B} in {@code A*X=B}.
     * @param aRankOriginal Original rank of {@code A} before any reshaping.
     * @return The shape of {@code X} in {@code A*X=B}.
     */
    protected Shape getOutputShape(T A, T B, int aRankOriginal) {
        int start = -(aRankOriginal - B.getRank());
        int stop = A.shape.dims.length;
        int[] dims = new int[stop-start];

        System.arraycopy(A.shape.dims, start, dims, 0, dims.length);

        return new Shape(true, dims);
    }


    /**
     * Ensures that aNumEntries==prod.
     * @param aNumEntries The total number of entries in the {@code A} tensor.
     * @param prod Product of all axis lengths in the output shape.
     */
    protected void checkSize(int aNumEntries, int prod) {
        ParameterChecks.assertEquals(aNumEntries, prod);
    }


    /**
     * Initializes matrix for equivalent linear matrix equation.
     * @param A Tensor to convert to matrix.
     * @param prod Product of all axis lengths in {@code A}.
     * @return A matrix with the same entries as tensor {@code A} with shape (prod, prod).
     */
    protected abstract U initMatrix(T A, int prod);


    /**
     * Initializes vector for equivalent linear matrix equation.
     * @param B Tensor to convert to vector.
     * @return Flattens tensor {@code B} and converts to a vector.
     */
    protected abstract V initVector(T B);


    /**
     * Wraps solution as a tensor and reshapes to the proper shape.
     * @param x Vector solution to matrix linear equation which is equivalent to the tensor equation {@code A*X=B}.
     * @param outputShape Shape for the solution tensor {@code X}.
     * @return The solution {@code X} to the linear tensor equation {@code A*X=B}.
     */
    protected abstract T wrap(V x, Shape outputShape);
}
