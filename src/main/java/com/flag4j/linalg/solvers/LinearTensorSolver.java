package com.flag4j.linalg.solvers;

import com.flag4j.core.TensorBase;

/**
 * This interface specifies methods which all linear tensor system solvers should implement. Solvers
 * may solve in an exact sense or in a least squares sense.
 * @param <T> Type of the tensors in the linear system.
 */
public interface LinearTensorSolver<T extends TensorBase<T, ?, ?, ?, ?, ?, ?>> {


    /**
     * Solves the linear tensor equation given by {@code A*X=B} for the tensor {@code X}. All indices of {@code X} are summed over in
     * the tensor product with the rightmost indices of {@code A} as if by
     * {@link com.flag4j.core.TensorExclusiveMixin#tensorDot(TensorBase, int)  A.tensorDot(X, X.getRank())}.
     * @param A Coefficient tensor in the linear system.
     * @param B Tensor of constants in the linear system.
     * @return The solution to {@code x} in the linear system {@code A*X=B}.
     */
    T solve(T A, T B);
}
