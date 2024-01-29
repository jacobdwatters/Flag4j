package com.flag4j.linalg.solvers;

import com.flag4j.CMatrix;
import com.flag4j.CTensor;
import com.flag4j.CVector;
import com.flag4j.Shape;


/**
 * Solver for solving a complex well determined linear tensor equation {@code A*X=B} in an exact sense.
 */
public class ComplexExactTensorSolver extends ExactTensorSolver<CTensor, CMatrix, CVector> {


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
     * @param A    Tensor to convert to matrix.
     * @param prod Product of all axis lengths in {@code A}.
     * @return A matrix with the same entries as tensor {@code A} with shape (prod, prod).
     */
    @Override
    protected CMatrix initMatrix(CTensor A, int prod) {
        return new CMatrix(prod, prod, A.entries);
    }


    /**
     * Initializes vector for equivalent linear matrix equation.
     *
     * @param B Tensor to convert to vector.
     * @return Flattens tensor {@code B} and converts to a vector.
     */
    @Override
    protected CVector initVector(CTensor B) {
        return new CVector(B.entries);
    }


    /**
     * Wraps solution as a tensor and reshapes to the proper shape.
     *
     * @param x           Vector solution to matrix linear equation which is equivalent to the tensor equation {@code A*X=B}.
     * @param outputShape Shape for the solution tensor {@code X}.
     * @return The solution {@code X} to the linear tensor equation {@code A*X=B}.
     */
    @Override
    protected CTensor wrap(CVector x, Shape outputShape) {
        return new CTensor(outputShape, x.entries);
    }
}
