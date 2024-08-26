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

package org.flag4j.operations_old.common;

import org.flag4j.arrays_old.dense.*;
import org.flag4j.arrays_old.sparse.*;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.TensorBase;
import org.flag4j.core.dense_base.ComplexDenseTensorBase;
import org.flag4j.core.dense_base.DenseTensorBase;
import org.flag4j.core.dense_base.RealDenseTensorBase;
import org.flag4j.core.sparse_base.SparseTensorBase;
import org.flag4j.operations_old.dense.complex.ComplexDenseEquals;
import org.flag4j.operations_old.dense.real.RealDenseEquals;
import org.flag4j.operations_old.dense.real_complex.RealComplexDenseEquals;
import org.flag4j.operations_old.dense_sparse.coo.complex.ComplexDenseSparseEquals;
import org.flag4j.operations_old.dense_sparse.coo.real.RealDenseSparseEquals;
import org.flag4j.operations_old.dense_sparse.coo.real_complex.RealComplexDenseSparseEquals;
import org.flag4j.operations_old.sparse.coo.complex.ComplexSparseEquals;
import org.flag4j.operations_old.sparse.coo.real.RealSparseEquals;
import org.flag4j.operations_old.sparse.coo.real_complex.RealComplexSparseEquals;
import org.flag4j.util.ErrorMessages;

import java.util.HashMap;
import java.util.function.BiFunction;

/**
 * Utility class for determining if arbitrary pairs of tensors are equal. Could be a real dense tensor and a sparse complex matrix,
 * if they have the same shape and entries, then they are considered equal.
 */
@Deprecated
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
            BiFunction<CooTensorOld,
                    SparseTensorBase<?, ?, ?, ?, ?, ?, ?>,
                    Boolean>> realSparseLookUp = new HashMap<>();
    private static final HashMap<String,
            BiFunction<CooCTensorOld,
                    SparseTensorBase<?, ?, ?, ?, ?, ?, ?>,
                    Boolean>> complexSparseLookUp = new HashMap<>();

    // Initialize algorithm lookup tables for the class.
    static {
        realDenseLookUp.put("CooTensorOld", (A, B) -> RealDenseSparseEquals.tensorEquals(A, (CooTensorOld) B));
        realDenseLookUp.put("CooCTensorOld", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, (CooCTensorOld) B));
//        realDenseLookUp.put("CooMatrixOld", (A, B) -> RealDenseSparseEquals.tensorEquals(A, ((CooMatrixOld) B).toTensor()));
//        realDenseLookUp.put("CooCMatrixOld", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, ((CooCMatrixOld) B).toTensor()));
//        realDenseLookUp.put("CsrMatrixOld", (A, B) -> RealDenseSparseEquals.tensorEquals(A, ((CsrMatrixOld) B).toTensor()));
//        realDenseLookUp.put("CsrCMatrixOld", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, ((CsrCMatrixOld) B).toTensor()));
//        realDenseLookUp.put("CooVectorOld", (A, B) -> RealDenseSparseEquals.tensorEquals(A, ((CooVectorOld) B).toTensor()));
//        realDenseLookUp.put("CooCVectorOld", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, ((CooCVectorOld) B).toTensor()));

        complexDenseLookUp.put("CooTensorOld", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, (CooTensorOld) B));
        complexDenseLookUp.put("CooCTensorOld", (A, B) -> ComplexDenseSparseEquals.tensorEquals(A, (CooCTensorOld) B));
//        complexDenseLookUp.put("CooMatrixOld", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, ((CooMatrixOld) B).toTensor()));
//        complexDenseLookUp.put("CooCMatrixOld", (A, B) -> ComplexDenseSparseEquals.tensorEquals(A, ((CooCMatrixOld) B).toTensor()));
//        complexDenseLookUp.put("CsrMatrixOld", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, ((CsrMatrixOld) B).toTensor()));
//        complexDenseLookUp.put("CsrCMatrixOld", (A, B) -> ComplexDenseSparseEquals.tensorEquals(A, ((CsrCMatrixOld) B).toTensor()));
//        complexDenseLookUp.put("CooVectorOld", (A, B) -> RealComplexDenseSparseEquals.tensorEquals(A, ((CooVectorOld) B).toTensor()));
//        complexDenseLookUp.put("CooCVectorOld", (A, B) -> ComplexDenseSparseEquals.tensorEquals(A, ((CooCVectorOld) B).toTensor()));

        realSparseLookUp.put("CooTensorOld", (A, B) -> RealSparseEquals.tensorEquals(A, (CooTensorOld) B));
        realSparseLookUp.put("CooCTensorOld", (A, B) -> RealComplexSparseEquals.tensorEquals(A, (CooCTensorOld) B));
//        realSparseLookUp.put("CooMatrixOld", (A, B) -> RealSparseEquals.tensorEquals(A, ((CooMatrixOld) B).toTensor()));
//        realSparseLookUp.put("CooCMatrixOld", (A, B) -> RealComplexSparseEquals.tensorEquals(A, ((CooCMatrixOld) B).toTensor()));
//        realSparseLookUp.put("CsrMatrixOld", (A, B) -> RealSparseEquals.tensorEquals(A, ((CsrMatrixOld) B).toTensor()));
//        realSparseLookUp.put("CsrCMatrixOld", (A, B) -> RealComplexSparseEquals.tensorEquals(A, ((CsrCMatrixOld) B).toTensor()));
//        realSparseLookUp.put("CooVectorOld", (A, B) -> RealSparseEquals.tensorEquals(A, ((CooVectorOld) B).toTensor()));
//        realSparseLookUp.put("CooCVectorOld", (A, B) -> RealComplexSparseEquals.tensorEquals(A, ((CooCVectorOld) B).toTensor()));

        complexSparseLookUp.put("CooTensorOld", (A, B) -> RealComplexSparseEquals.tensorEquals((CooTensorOld) B, A));
        complexSparseLookUp.put("CooCTensorOld", (A, B) -> ComplexSparseEquals.tensorEquals(A, (CooCTensorOld) B));
//        complexSparseLookUp.put("CooMatrixOld", (A, B) -> RealComplexSparseEquals.tensorEquals(((CooMatrixOld) B).toTensor(), A));
//        complexSparseLookUp.put("CooCMatrixOld", (A, B) -> ComplexSparseEquals.tensorEquals(A, ((CooCMatrixOld) B).toTensor()));
//        complexSparseLookUp.put("CsrMatrixOld", (A, B) -> RealComplexSparseEquals.tensorEquals(((CsrMatrixOld) B).toTensor(), A));
//        complexSparseLookUp.put("CsrCMatrixOld", (A, B) -> ComplexSparseEquals.tensorEquals(A, ((CsrCMatrixOld) B).toTensor()));
//        complexSparseLookUp.put("CooVectorOld", (A, B) -> RealComplexSparseEquals.tensorEquals(((CooVectorOld) B).toTensor(), A));
//        complexSparseLookUp.put("CooCVectorOld", (A, B) -> ComplexSparseEquals.tensorEquals(A, ((CooCVectorOld) B).toTensor()));
    }

    private TensorEquals() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
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
            if(A instanceof CooTensorOld)
                return generalEquals((CooTensorOld) A, (SparseTensorBase<?, ?, ?, ?, ?, ?, ?>) B);
            else
                return generalEquals((CooCTensorOld) A, (SparseTensorBase<?, ?, ?, ?, ?, ?, ?>) B);

        } else if(A instanceof SparseTensorBase) {
            if(A instanceof CooTensorOld)
                return generalEquals((DenseTensorBase<?, ?, ?, ?, ?>) B, (CooTensorOld) A);
            else
                return generalEquals((DenseTensorBase<?, ?, ?, ?, ?>) B, (CooCTensorOld) A);

        } else if(B instanceof SparseTensorBase) {
            if(B instanceof CooTensorOld)
                return generalEquals((DenseTensorBase<?, ?, ?, ?, ?>) A, (CooTensorOld) B);
            else
                return generalEquals((DenseTensorBase<?, ?, ?, ?, ?>) A, (CooCTensorOld) B);
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
        if(src instanceof TensorOld || src instanceof CooTensorOld || src instanceof CTensorOld || src instanceof CooCTensorOld)
            return src;

        if(src instanceof VectorOld)
            src = ((VectorOld) src).toTensor();
        else if(src instanceof CVectorOld)
            src = ((CVectorOld) src).toTensor();
        else if(src instanceof MatrixOld)
            src = ((MatrixOld) src).toTensor();
        else if(src instanceof CMatrixOld)
            src = ((CMatrixOld) src).toTensor();
        else if(src instanceof CooVectorOld)
            src = ((CooVectorOld) src).toTensor();
        else if(src instanceof CooCVectorOld)
            src = ((CooCVectorOld) src).toTensor();
        else if(src instanceof CooMatrixOld)
            src = ((CooMatrixOld) src).toTensor();
        else if(src instanceof CooCMatrixOld)
            src = ((CooCMatrixOld) src).toTensor();
        else if(src instanceof CsrMatrixOld)
            src = ((CsrMatrixOld) src).toTensor();
        else if(src instanceof CsrCMatrixOld)
            src = ((CsrCMatrixOld) src).toTensor();

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
     * Checks if a real dense tensor is equal to any sparse tensor (including {@link CooMatrixOld}, {@link CsrMatrixOld},
     * {@link CooCMatrixOld}, {@link CsrCMatrixOld}, {@link CooVectorOld}, etc.).
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
     * Checks if a dense tensor is equal to any sparse tensor (including {@link CooMatrixOld}, {@link CsrMatrixOld},
     * {@link CooCMatrixOld}, {@link CsrCMatrixOld}, {@link CooVectorOld}, etc.).
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
     * Checks if a real dense tensor is equal to any sparse tensor (including {@link CooMatrixOld}, {@link CsrMatrixOld},
     * {@link CooCMatrixOld}, {@link CsrCMatrixOld}, {@link CooVectorOld}, etc.).
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
     * Checks if a complex dense tensor is equal to any sparse tensor (including {@link CooMatrixOld}, {@link CsrMatrixOld},
     * {@link CooCMatrixOld}, {@link CsrCMatrixOld}, {@link CooVectorOld}, etc.).
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
     * Checks if a real dense tensor is equal to any sparse tensor (including {@link CooMatrixOld}, {@link CsrMatrixOld},
     * {@link CooCMatrixOld}, {@link CsrCMatrixOld}, {@link CooVectorOld}, etc.).
     * @param A Real dense tensor.
     * @param B Sparse tensor.
     * @return True if the two matrices are element-wise equivalent.
     */
    private static boolean generalEquals(CooTensorOld A, SparseTensorBase<?, ?, ?, ?, ?, ?, ?> B) {
        String type = B.getClass().getSimpleName();

        if(!TensorEquals.realSparseLookUp.containsKey(type))
            throw new IllegalArgumentException("Unsupported tensor type: " + B.getClass());

        return TensorEquals.realSparseLookUp.get(type).apply(A, B);
    }


    /**
     * Checks if a real dense tensor is equal to any sparse tensor (including {@link CooMatrixOld}, {@link CsrMatrixOld},
     * {@link CooCMatrixOld}, {@link CsrCMatrixOld}, {@link CooVectorOld}, etc.).
     * @param A Real dense tensor.
     * @param B Sparse tensor.
     * @return True if the two matrices are element-wise equivalent.
     */
    private static boolean generalEquals(CooCTensorOld A, SparseTensorBase<?, ?, ?, ?, ?, ?, ?> B) {
        String type = B.getClass().getSimpleName();

        if(!TensorEquals.complexSparseLookUp.containsKey(type))
            throw new IllegalArgumentException("Unsupported tensor type: " + B.getClass());

        return TensorEquals.complexSparseLookUp.get(type).apply(A, B);
    }
}
