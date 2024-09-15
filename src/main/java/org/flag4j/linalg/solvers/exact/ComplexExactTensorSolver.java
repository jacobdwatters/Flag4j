package org.flag4j.linalg.solvers.exact;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.CVector;


/**
 * Solver for solving a complex well determined linear tensor equation A*X=B in an exact sense.
 */
public class ComplexExactTensorSolver extends ExactTensorSolver<CTensor, CMatrix, CVector> {

    /**
     * Creates an exact tensor solver for solving a well determined linear tensor equation A*X=B
     * for X in an exact sense.
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
     * @param x           Vector solution to matrix linear equation which is equivalent to the tensor equation A*X=B.
     * @param outputShape Shape for the solution tensor X.
     * @return The solution X to the linear tensor equation A*X=B.
     */
    @Override
    protected CTensor wrap(CVector x, Shape outputShape) {
        return new CTensor(outputShape, x.entries);
    }
}
