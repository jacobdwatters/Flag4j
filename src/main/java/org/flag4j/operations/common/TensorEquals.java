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

package org.flag4j.operations.common;

import org.flag4j.arrays_old.dense.*;
import org.flag4j.arrays_old.sparse.*;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.TensorBase;
import org.flag4j.core.dense_base.ComplexDenseTensorBase;
import org.flag4j.core.dense_base.DenseTensorBase;
import org.flag4j.core.dense_base.RealDenseTensorBase;
import org.flag4j.core.sparse_base.SparseTensorBase;
import org.flag4j.operations.dense.complex.ComplexDenseEquals;
import org.flag4j.operations.dense.real.RealDenseEquals;
import org.flag4j.operations.dense.real_complex.RealComplexDenseEquals;
import org.flag4j.operations.dense_sparse.coo.complex.ComplexDenseSparseEquals;
import org.flag4j.operations.dense_sparse.coo.real.RealDenseSparseEquals;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseEquals;
import org.flag4j.operations.sparse.coo.complex.ComplexSparseEquals;
import org.flag4j.operations.sparse.coo.real.RealSparseEquals;
import org.flag4j.operations.sparse.coo.real_complex.RealComplexSparseEquals;
import org.flag4j.util.ErrorMessages;

import java.util.HashMap;
import java.util.function.BiFunction;

/**
 * Utility class for determining if arbitrary pairs of tensors are equal. Could be a real dense tensor and a sparse complex matrix,
 * if they have the same shape and entries, then they are considered equal.
 */
public final class TensorEquals {

    private static final HashMap<String,
            BiFunction<TensorOld,
                    SparseTensorBase<?, ?, ?, ?, ?, ?, ?>,
                    Boolean>> realDenseLookUp = new HashMap<>();
    private static final HashMap<String,
            BiFunction<CTensorOld,
                    SparseTensorBase<?, ?, ?, ?, ?, ?, ?>,
                    Boolean>> complexDenseLookUp = new HashMap<>();
    private static final HashMap<String,
            BiFunction<CooTensor,
                    SparseTensorBase<?, ?, ?, ?, ?, ?, ?>,
                    Boolean>> realSparseLookUp = new HashMap<>();
    private static final HashMap<String,
            BiFunction<CooCTensor,
                    SparseTensorBase<?, ?, ?, ?, ?, ?, ?>,
                    Boolean>> complexSparseLookUp = new HashMap<>();

    // Initialize algorithm lookup tables for the class.
    static {
        realDenseLookUp.put("CooTensor", (A, B) -> RealDenseSparseEquals.tensorEquals(A, (CooTensor) B));
        realDenseLookUp.put("CooCTensor", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, (CooCTensor) B));
//        realDenseLookUp.put("CooMatrix", (A, B) -> RealDenseSparseEquals.tensorEquals(A, ((CooMatrix) B).toTensor()));
//        realDenseLookUp.put("CooCMatrix", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, ((CooCMatrix) B).toTensor()));
//        realDenseLookUp.put("CsrMatrix", (A, B) -> RealDenseSparseEquals.tensorEquals(A, ((CsrMatrix) B).toTensor()));
//        realDenseLookUp.put("CsrCMatrix", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, ((CsrCMatrix) B).toTensor()));
//        realDenseLookUp.put("CooVector", (A, B) -> RealDenseSparseEquals.tensorEquals(A, ((CooVector) B).toTensor()));
//        realDenseLookUp.put("CooCVector", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, ((CooCVector) B).toTensor()));

        complexDenseLookUp.put("CooTensor", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, (CooTensor) B));
        complexDenseLookUp.put("CooCTensor", (A, B) -> ComplexDenseSparseEquals.tensorEquals(A, (CooCTensor) B));
//        complexDenseLookUp.put("CooMatrix", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, ((CooMatrix) B).toTensor()));
//        complexDenseLookUp.put("CooCMatrix", (A, B) -> ComplexDenseSparseEquals.tensorEquals(A, ((CooCMatrix) B).toTensor()));
//        complexDenseLookUp.put("CsrMatrix", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, ((CsrMatrix) B).toTensor()));
//        complexDenseLookUp.put("CsrCMatrix", (A, B) -> ComplexDenseSparseEquals.tensorEquals(A, ((CsrCMatrix) B).toTensor()));
//        complexDenseLookUp.put("CooVector", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, ((CooVector) B).toTensor()));
//        complexDenseLookUp.put("CooCVector", (A, B) -> ComplexDenseSparseEquals.tensorEquals(A, ((CooCVector) B).toTensor()));

        realSparseLookUp.put("CooTensor", (A, B) -> RealSparseEquals.tensorEquals(A, (CooTensor) B));
        realSparseLookUp.put("CooCTensor", (A, B) -> RealComplexSparseEquals.tensorEquals(A, (CooCTensor) B));
//        realSparseLookUp.put("CooMatrix", (A, B) -> RealSparseEquals.tensorEquals(A, ((CooMatrix) B).toTensor()));
//        realSparseLookUp.put("CooCMatrix", (A, B) -> RealComplexSparseEquals.tensorEquals(A, ((CooCMatrix) B).toTensor()));
//        realSparseLookUp.put("CsrMatrix", (A, B) -> RealSparseEquals.tensorEquals(A, ((CsrMatrix) B).toTensor()));
//        realSparseLookUp.put("CsrCMatrix", (A, B) -> RealComplexSparseEquals.tensorEquals(A, ((CsrCMatrix) B).toTensor()));
//        realSparseLookUp.put("CooVector", (A, B) -> RealSparseEquals.tensorEquals(A, ((CooVector) B).toTensor()));
//        realSparseLookUp.put("CooCVector", (A, B) -> RealComplexSparseEquals.tensorEquals(A, ((CooCVector) B).toTensor()));

        complexSparseLookUp.put("CooTensor", (A, B) -> RealComplexSparseEquals.tensorEquals((CooTensor) B, A));
        complexSparseLookUp.put("CooCTensor", (A, B) -> ComplexSparseEquals.tensorEquals(A, (CooCTensor) B));
//        complexSparseLookUp.put("CooMatrix", (A, B) -> RealComplexSparseEquals.tensorEquals(((CooMatrix) B).toTensor(), A));
//        complexSparseLookUp.put("CooCMatrix", (A, B) -> ComplexSparseEquals.tensorEquals(A, ((CooCMatrix) B).toTensor()));
//        complexSparseLookUp.put("CsrMatrix", (A, B) -> RealComplexSparseEquals.tensorEquals(((CsrMatrix) B).toTensor(), A));
//        complexSparseLookUp.put("CsrCMatrix", (A, B) -> ComplexSparseEquals.tensorEquals(A, ((CsrCMatrix) B).toTensor()));
//        complexSparseLookUp.put("CooVector", (A, B) -> RealComplexSparseEquals.tensorEquals(((CooVector) B).toTensor(), A));
//        complexSparseLookUp.put("CooCVector", (A, B) -> ComplexSparseEquals.tensorEquals(A, ((CooCVector) B).toTensor()));
    }

    private TensorEquals() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks if two arbitrary tensors are equal regardless implementation (e.g. any combination of real, complex, dense and sparse).
     * Tensors do not need to be the same type and for the purposes of this method, matrices and vectors are considered tensors.
     *
     * @param A First tensor in equality comparison.
     * @param B Second tensor in equality comparison.
     * @return True if both tensors have the same shape and are element-wise equal.
     */
    public static boolean generalEquals(TensorBase<?, ?, ?, ?, ?, ?, ?> A, TensorBase<?, ?, ?, ?, ?, ?, ?> B) {
        // Check for quick returns.
        if(A == B) return true;
        if(A == null || B ==null) return false;
        if(!A.shape.equals(B.shape)) return false;

        // Ensure both objects are explicit tensors.
        A = ensureTensor(A);
        B = ensureTensor(B);

        if(A instanceof SparseTensorBase && B instanceof SparseTensorBase) {
            if(A instanceof CooTensor)
                return generalEquals((CooTensor) A, (SparseTensorBase<?, ?, ?, ?, ?, ?, ?>) B);
            else
                return generalEquals((CooCTensor) A, (SparseTensorBase<?, ?, ?, ?, ?, ?, ?>) B);

        } else if(A instanceof SparseTensorBase) {
            if(A instanceof CooTensor)
                return generalEquals((DenseTensorBase<?, ?, ?, ?, ?>) B, (CooTensor) A);
            else
                return generalEquals((DenseTensorBase<?, ?, ?, ?, ?>) B, (CooCTensor) A);

        } else if(B instanceof SparseTensorBase) {
            if(B instanceof CooTensor)
                return generalEquals((DenseTensorBase<?, ?, ?, ?, ?>) A, (CooTensor) B);
            else
                return generalEquals((DenseTensorBase<?, ?, ?, ?, ?>) A, (CooCTensor) B);
        } else {
            // Then both tensors are dense.
            if(A instanceof TensorOld)
                return generalEquals((TensorOld) A, (DenseTensorBase<?, ?, ?, ?, ?>) B);
            else
                return generalEquals((CTensorOld) A, (DenseTensorBase<?, ?, ?, ?, ?>) B);
        }
    }

    /**
     * Converts a tensor to an explicit tensor if it is a vector or matrix.
     * @param src TensorOld to convert to an explicit tensor.
     * @return Returns {@code src} if it is already a tensor. Otherwise, i.e. {@code src} is a matrix or vector, returns an explicit
     * tensor equivalent to {@code src}.
     */
    private static TensorBase<?, ?, ?, ?, ?, ?, ?> ensureTensor(TensorBase<?, ?, ?, ?, ?, ?, ?> src) {
        // Check for quick return.
        if(src instanceof TensorOld || src instanceof CooTensor || src instanceof CTensorOld || src instanceof CooCTensor)
            return src;

        if(src instanceof VectorOld)
            src = ((VectorOld) src).toTensor();
        else if(src instanceof CVectorOld)
            src = ((CVectorOld) src).toTensor();
        else if(src instanceof MatrixOld)
            src = ((MatrixOld) src).toTensor();
        else if(src instanceof CMatrixOld)
            src = ((CMatrixOld) src).toTensor();
        else if(src instanceof CooVector)
            src = ((CooVector) src).toTensor();
        else if(src instanceof CooCVector)
            src = ((CooCVector) src).toTensor();
        else if(src instanceof CooMatrix)
            src = ((CooMatrix) src).toTensor();
        else if(src instanceof CooCMatrix)
            src = ((CooCMatrix) src).toTensor();
        else if(src instanceof CsrMatrix)
            src = ((CsrMatrix) src).toTensor();
        else if(src instanceof CsrCMatrix)
            src = ((CsrCMatrix) src).toTensor();

        return src;
    }


    /**
     * Checks if a real dense tensor is equal to any dense tensor (including {@link MatrixOld}, {@link CMatrixOld}, {@link VectorOld}, etc.).
     * @param A Real dense tensor.
     * @param B Sparse tensor.
     * @return True if the two matrices are element-wise equivalent.
     */
    private static boolean generalEquals(TensorOld A, DenseTensorBase<?, ?, ?, ?, ?> B) {
        if(B instanceof RealDenseTensorBase) {
            return RealDenseEquals.tensorEquals(A.entries, A.shape, (double[]) B.entries, B.shape);
        } else if(B instanceof ComplexDenseTensorBase) {
            return RealComplexDenseEquals.tensorEquals(A.entries, A.shape, (CNumber[]) B.entries, B.shape);
        } else {
            throw new IllegalArgumentException("Unsupported tensor type: " + B.getClass());
        }
    }


    /**
     * Checks if a real dense tensor is equal to any sparse tensor (including {@link CooMatrix}, {@link CsrMatrix},
     * {@link CooCMatrix}, {@link CsrCMatrix}, {@link CooVector}, etc.).
     * @param A Real dense tensor.
     * @param B Sparse tensor.
     * @return True if the two matrices are element-wise equivalent.
     */
    private static boolean generalEquals(CTensorOld A, DenseTensorBase<?, ?, ?, ?, ?> B) {
        if(B instanceof RealDenseTensorBase) {
            return RealComplexDenseEquals.tensorEquals((double[]) B.entries, B.shape, A.entries, A.shape);
        } else if(B instanceof ComplexDenseTensorBase) {
            return ComplexDenseEquals.tensorEquals(A.entries, A.shape, (CNumber[]) B.entries, B.shape);
        } else {
            throw new IllegalArgumentException("Unsupported tensor type: " + B.getClass());
        }
    }


    /**
     * Checks if a dense tensor is equal to any sparse tensor (including {@link CooMatrix}, {@link CsrMatrix},
     * {@link CooCMatrix}, {@link CsrCMatrix}, {@link CooVector}, etc.).
     * @param A Real dense tensor.
     * @param B Sparse tensor.
     * @return True if the two matrices are element-wise equivalent.
     */
    private static boolean generalEquals(DenseTensorBase<?, ?, ?, ?, ?> A, SparseTensorBase<?, ?, ?, ?, ?, ?, ?> B) {
        if(A instanceof TensorOld)
            return generalEquals((TensorOld) A, B);
        else if(A instanceof CTensorOld)
            return generalEquals((CTensorOld) A, B);
        else
            throw new IllegalArgumentException("Unsupported tensor type: " + A.getClass());
    }


    /**
     * Checks if a real dense tensor is equal to any sparse tensor (including {@link CooMatrix}, {@link CsrMatrix},
     * {@link CooCMatrix}, {@link CsrCMatrix}, {@link CooVector}, etc.).
     * @param A Real dense tensor.
     * @param B Sparse tensor.
     * @return True if the two matrices are element-wise equivalent.
     */
    private static boolean generalEquals(TensorOld A, SparseTensorBase<?, ?, ?, ?, ?, ?, ?> B) {
        String type = B.getClass().getSimpleName();

        if(!TensorEquals.realDenseLookUp.containsKey(type))
            throw new IllegalArgumentException("Unsupported tensor type: " + B.getClass());

        // Apply the appropriate tensor equality implementation.
        return TensorEquals.realDenseLookUp.get(type).apply(A, B);
    }


    /**
     * Checks if a complex dense tensor is equal to any sparse tensor (including {@link CooMatrix}, {@link CsrMatrix},
     * {@link CooCMatrix}, {@link CsrCMatrix}, {@link CooVector}, etc.).
     * @param A Real dense tensor.
     * @param B Sparse tensor.
     * @return True if the two matrices are element-wise equivalent.
     */
    private static boolean generalEquals(CTensorOld A, SparseTensorBase<?, ?, ?, ?, ?, ?, ?> B) {
        String type = B.getClass().getSimpleName();

        if(!TensorEquals.complexDenseLookUp.containsKey(type))
            throw new IllegalArgumentException("Unsupported tensor type: " + B.getClass());

        // Apply the appropriate tensor equality implementation.
        return TensorEquals.complexDenseLookUp.get(type).apply(A, B);
    }


    /**
     * Checks if a real dense tensor is equal to any sparse tensor (including {@link CooMatrix}, {@link CsrMatrix},
     * {@link CooCMatrix}, {@link CsrCMatrix}, {@link CooVector}, etc.).
     * @param A Real dense tensor.
     * @param B Sparse tensor.
     * @return True if the two matrices are element-wise equivalent.
     */
    private static boolean generalEquals(CooTensor A, SparseTensorBase<?, ?, ?, ?, ?, ?, ?> B) {
        String type = B.getClass().getSimpleName();

        if(!TensorEquals.realSparseLookUp.containsKey(type))
            throw new IllegalArgumentException("Unsupported tensor type: " + B.getClass());

        return TensorEquals.realSparseLookUp.get(type).apply(A, B);
    }


    /**
     * Checks if a real dense tensor is equal to any sparse tensor (including {@link CooMatrix}, {@link CsrMatrix},
     * {@link CooCMatrix}, {@link CsrCMatrix}, {@link CooVector}, etc.).
     * @param A Real dense tensor.
     * @param B Sparse tensor.
     * @return True if the two matrices are element-wise equivalent.
     */
    private static boolean generalEquals(CooCTensor A, SparseTensorBase<?, ?, ?, ?, ?, ?, ?> B) {
        String type = B.getClass().getSimpleName();

        if(!TensorEquals.complexSparseLookUp.containsKey(type))
            throw new IllegalArgumentException("Unsupported tensor type: " + B.getClass());

        return TensorEquals.complexSparseLookUp.get(type).apply(A, B);
    }
}
