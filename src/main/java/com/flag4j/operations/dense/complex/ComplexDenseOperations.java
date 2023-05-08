/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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

package com.flag4j.operations.dense.complex;

import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.Axis2D;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import static com.flag4j.operations.common.complex.AggregateComplex.maxAbs;
import static com.flag4j.operations.common.real.AggregateReal.maxAbs;


/**
 * This class provides low level methods for computing operations on dense complex tensors.
 */
public final class ComplexDenseOperations {

    private ComplexDenseOperations() {
        // Hide constructor
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the element-wise addition of two tensors.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return The element wise addition of two tensors.
     * @throws IllegalArgumentException If entry arrays are not the same size.
     */
    public static CNumber[] add(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        ParameterChecks.assertEqualShape(shape1, shape2);

        CNumber[] sum = new CNumber[src1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = src1[i].add(src2[i]);
        }

        return sum;
    }


    /**
     * Adds a scalar value to all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar addition of the two parameters.
     */
    public static CNumber[] add(CNumber[] src1, CNumber a) {
        CNumber[] sum = new CNumber[src1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = src1[i].add(a);
        }

        return sum;
    }


    /**
     * Adds a scalar value to all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar addition of the two parameters.
     */
    public static CNumber[] add(double[] src1, CNumber a) {
        CNumber[] sum = new CNumber[src1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = a.add(src1[i]);
        }

        return sum;
    }


    /**
     * Subtracts a scalar value to all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar addition of the two parameters.
     */
    public static CNumber[] sub(double[] src1, CNumber a) {
        CNumber[] sum = new CNumber[src1.length];
        ArrayUtils.fillZeros(sum);

        for(int i=0; i<sum.length; i++) {
            sum[i].re = src1[i]-a.re;
            sum[i].im = -a.im;
        }

        return sum;
    }


    /**
     * Computes the element-wise subtraction of two tensors.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return The element wise subtraction of two tensors.
     * @throws IllegalArgumentException If entry arrays are not the same size.
     */
    public static CNumber[] sub(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        ParameterChecks.assertEqualShape(shape1, shape2);

        CNumber[] diff = new CNumber[src1.length];

        for(int i=0; i<diff.length; i++) {
            diff[i] = src1[i].sub(src2[i]);
        }

        return diff;
    }


    /**
     * Subtracts a scalar value from all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar subtraction of the two parameters.
     */
    public static CNumber[] sub(CNumber[] src1, CNumber a) {
        CNumber[] sum = new CNumber[src1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = src1[i].sub(a);
        }

        return sum;
    }


    /**
     * Computes element-wise subtraction between tensors and stores the result in the first tensor.
     * @param src1 First tensor in subtraction. Also, where the result will be stored.
     * @param shape1 Shape of first tensor.
     * @param src2 Second tensor in the subtraction.
     * @param shape2 Shape of second tensor.
     */
    public static void subEq(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        ParameterChecks.assertEqualShape(shape1, shape2);

        for(int i=0; i<src1.length; i++) {
            src1[i].subEq(src2[i]);
        }
    }


    /**
     * Subtracts a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src Tensor in subtraction. Also, where the result will be stored.
     * @param b Scalar to subtract.
     */
    public static void subEq(CNumber[] src, CNumber b) {
        for(CNumber cNumber : src) {
            cNumber.subEq(b);
        }
    }


    /**
     * Computes element-wise addition between tensors and stores the result in the first tensor.
     * @param src1 First tensor in addition. Also, where the result will be stored.
     * @param shape1 Shape of first tensor.
     * @param src2 Second tensor in the addition.
     * @param shape2 Shape of second tensor.
     */
    public static void addEq(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        ParameterChecks.assertEqualShape(shape1, shape2);

        for(int i=0; i<src1.length; i++) {
            src1[i].addEq(src2[i]);
        }
    }


    /**
     * Adds a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src Tensor in addition. Also, where the result will be stored.
     * @param b Scalar to add.
     */
    public static void addEq(CNumber[] src, CNumber b) {
        for (CNumber cNumber : src) {
            cNumber.addEq(b);
        }
    }


    /**
     * Multiplies all entries in a tensor.
     * @param src Entries of tensor to compute product of.
     * @return The product of all src in the tensor.
     */
    public static CNumber prod(CNumber[] src) {
        CNumber product;

        if(src.length>0) {
            product = new CNumber(1);
            for(CNumber value : src) {
               product = product.mult(value);
            }
        } else {
            product = new CNumber();
        }

        return product;
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param a Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static CNumber[] scalMult(CNumber[] entries, CNumber a) {
        CNumber[] product = new CNumber[entries.length];

        for(int i=0; i<product.length; i++) {
            product[i] = entries[i].mult(a);
        }

        return product;
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param a Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static CNumber[] scalMult(CNumber[] entries, double a) {
        CNumber[] product = new CNumber[entries.length];

        for(int i=0; i<product.length; i++) {
            product[i] = entries[i].mult(a);
        }

        return product;
    }


    /**
     * Computes the scalar division of a tensor.
     * @param entries Entries of the tensor.
     * @param divisor Scalar value to divide by.
     * @return The scalar division of the tensor.
     */
    public static CNumber[] scalDiv(CNumber[] entries, CNumber divisor) {
        CNumber[] quotient = new CNumber[entries.length];

        for(int i=0; i<quotient.length; i++) {
            quotient[i] = entries[i].div(divisor);
        }

        return quotient;
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise reciprocals of the tensor.
     */
    public static CNumber[] recip(CNumber[] src) {
        CNumber[] receps = new CNumber[src.length];

        for(int i=0; i<receps.length; i++) {
            receps[i] = src[i].addInv();
        }

        return receps;
    }


    /**
     * Compute the L<sub>p, q</sub> norm of a matrix.
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @param p First parameter in L<sub>p, q</sub> norm.
     * @param q Second parameter in L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of the matrix.
     * @throws IllegalArgumentException If {@code p} or {@code q} is less than 1.
     */
    public static double matrixNormLpq(CNumber[] src, Shape shape, double p, double q) {
        ParameterChecks.assertGreaterEq(1, p, q);

        double norm = 0;
        double colSum;
        int rows = shape.dims[Axis2D.row()];
        int cols = shape.dims[Axis2D.col()];

        for(int j=0; j<cols; j++) {
            colSum = 0;
            for(int i=0; i<rows; i++) {
                colSum += (Math.pow(src[i*cols + j].magAsDouble(), p));
            }
            norm += Math.pow(colSum, q/p);
        }

        return Math.pow(norm, 1/q);
    }


    /**
     * Compute the L<sub>p</sub> norm of a matrix. This is equivalent to passing {@code q=1} to
     * {@link ComplexDenseOperations#matrixNormLpq(CNumber[], Shape, double, double)}
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @param p Parameter in L<sub>p</sub> norm.
     * @return The L<sub>p</sub> norm of the matrix.
     * @throws IllegalArgumentException If {@code p} is less than 1.
     */
    public static double matrixNormLp(CNumber[] src, Shape shape, double p) {
        ParameterChecks.assertGreaterEq(1, p);

        double norm = 0;
        double colSum;
        int rows = shape.dims[Axis2D.row()];
        int cols = shape.dims[Axis2D.col()];

        for(int j=0; j<cols; j++) {
            colSum=0;
            for(int i=0; i<rows; i++) {
                colSum += Math.pow(src[i*cols + j].magAsDouble(), p);
            }

            norm += Math.pow(colSum, 1.0/p);
        }

        return norm;
    }


    /**
     * Compute the L<sub>2</sub> norm of a matrix. This is equivalent to passing {@code q=1} to
     * {@link ComplexDenseOperations#matrixNormLpq(CNumber[], Shape, double, double)}
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @return The L<sub>2</sub> norm of the matrix.
     */
    public static double matrixNormL2(CNumber[] src, Shape shape) {
        double norm = 0;
        int rows = shape.dims[Axis2D.row()];
        int cols = shape.dims[Axis2D.col()];

        double colSum;

        for(int j=0; j<cols; j++) {
            colSum = 0;
            for(int i=0; i<rows; i++) {
                colSum += Math.pow(src[i*cols + j].magAsDouble(), 2);
            }
            norm += Math.sqrt(colSum);
        }

        return norm;
    }


    /**
     * Computes the infinity/maximum norm of a matrix. That is, the maximum value in this matrix.
     * @param src Entries of the matrix.
     * @return The infinity norm of the matrix.
     */
    public static double matrixMaxNorm(CNumber[] src) {
        return maxAbs(src);
    }


    /**
     * Computes the infinity/maximum norm of a matrix. That is, the maximum absolute value in this matrix.
     * @param src Entries of the matrix.
     * @return The infinity norm of the matrix.
     */
    public static double matrixInfNorm(CNumber[] src, Shape shape) {
        int rows = shape.dims[Axis2D.row()];
        int cols = shape.dims[Axis2D.col()];
        double[] rowSums = new double[rows];

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                rowSums[i] += src[i*cols + j].magAsDouble();
            }
        }

        return maxAbs(rowSums);
    }


    /**
     * Computes the L<sub>2</sub> norm of a tensor (i.e. the Frobenius norm).
     * @param src Entries of the tensor.
     * @return The L<sub>2</sub> norm of the tensor.
     */
    public static double tensorNormL2(CNumber[] src) {
        double norm = 0;

        for(CNumber cNumber : src) {
            norm += CNumber.pow(cNumber, 2).magAsDouble();
        }

        return Math.sqrt(norm);
    }


    /**
     * Computes the L<sub>p</sub> norm of a tensor (i.e. the Frobenius norm).
     * @param src Entries of the tensor.
     * @param p The {@code p} parameter of the L<sub>p</sub> norm.
     * @return The L<sub>p</sub> norm of the tensor.
     */
    public static double tensorNormLp(CNumber[] src, double p) {
        double norm = 0;

        for(CNumber cNumber : src) {
            norm += CNumber.pow(cNumber, p).magAsDouble();
        }

        return Math.pow(norm, 1.0/p);
    }
}
