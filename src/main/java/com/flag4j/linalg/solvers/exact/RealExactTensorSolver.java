package com.flag4j.linalg.solvers.exact;


import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.Tensor;
import com.flag4j.Vector;

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
