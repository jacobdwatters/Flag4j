/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.linalg.transformations;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CMatrix;
import org.flag4j.dense.CVector;
import org.flag4j.dense.Matrix;
import org.flag4j.dense.Vector;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.operations.common.complex.ComplexOperations;
import org.flag4j.operations.common.real.RealOperations;
import org.flag4j.util.ErrorMessages;

/**
 * This class contains methods for computing real or complex Householder reflectors (also known as elementary reflectors).
 * A Householder reflector is a transformation matrix which reflects a vector about a hyperplane containing the origin.
 */
public class Householder {

    private Householder() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the Householder reflector which describes a reflection through a hyperplane containing the origin which
     * is normal to the specified {@code normal} vector.
     * @param normal The vector normal to the plane the Householder reflector will reflect through.
     * @return A transformation matrix which describes a reflection through a plane containing the origin with the
     * specified {@code normal} vector, i.e. a Householder reflector.
     */
    public static Matrix getReflector(Vector normal) {
        Vector v;

        double signedNorm = -Math.copySign(VectorNorms.norm(normal), normal.entries[0]);
        v = normal.div(normal.entries[0] - signedNorm);
        v.entries[0] = 1.0;

        // Create Householder matrix
        Matrix H = v.outer(v).mult(-2.0/v.inner(v));

        int step = H.numCols+1;
        for(int i=0; i<H.entries.length; i+=step) {
            H.entries[i] = 1 + H.entries[i];
        }

        return H;
    }


    /**
     * Computes the vector {@code v} in of a Householder matrix {@code H=I-2vv}<sup>T</sup> where {@code H} is a
     * transformation matrix which reflects a vector across the plane normal to {@code normal}.
     *
     * <p>This method may be used in conjunction with {@link #leftMultReflector(Matrix, Vector, double, int, int, int)} and
     * {@link #rightMultReflector(Matrix, Vector, double, int, int, int)} to efficiently apply reflectors. Doing this is O(n^2)
     * while forming the full Householder matrix and performing matrix multiplication is O(n^3)</p>
     *
     * @param normal Vector normal to the plane which {@code H} reflects across.
     * @return The vector {@code v} in of a Householder matrix {@code H=I-2vv}<sup>T</sup> which reflects across a plane
     * normal to {@code normal}.
     */
    public static Vector getVector(Vector normal) {
        double normX = VectorNorms.norm(normal);
        double x1 = normal.entries[0];
        normX = (x1 >= 0) ? -normX : normX;
        double v1 = x1 - normX;

        // Initialize v with norm and set first element
        Vector v = normal.copy();
        v.entries[0] = v1;

        // Compute norm of v noting that it only differs from normal by the first element.
        double normV = Math.sqrt(normX*normX - x1*x1 + v1*v1);

        for(int i=0; i<v.entries.length; i++)
            v.entries[i] /= normV; // Normalize v to make it a unit vector

        return v;
    }


    /**
     * Computes the Householder reflector which describes a reflection through a hyperplane containing the origin which
     * is normal to the specified {@code normal} vector.
     *
     * <p>This method may be used in conjunction with {@link #leftMultReflector(CMatrix, CVector, CNumber, int, int, int)} and
     * {@link #rightMultReflector(CMatrix, CVector, CNumber, int, int, int)} to efficiently apply reflectors. Doing this is O(n^2)
     * while forming the full Householder matrix and performing matrix multiplication is O(n^3)</p>
     *
     * @param normal The vector normal to the plane the Householder reflector will reflect through\.
     * @return A transformation matrix which describes a reflection through a plane containing the origin with the
     * specified {@code normal} vector, i.e. a Householder reflector.
     */
    public static CMatrix getReflector(CVector normal) {
        CVector v;

        // Compute signed norm using modified sgn function.
        CNumber signedNorm = normal.entries[0].equals(0) ?
                new CNumber(-VectorNorms.norm(normal)) : CNumber.sgn(normal.entries[0]).mult(-VectorNorms.norm(normal));

        v = normal.div(normal.entries[0].sub(signedNorm));
        v.entries[0] = new CNumber(1);

        // Create projection matrix
        CMatrix P = v.outer(v).mult(-2.0/v.innerSelf());

        int step = P.numCols+1;
        for(int i=0; i<P.entries.length; i+=step) {
            P.entries[i].addEq(1.0);
        }

        return P;
    }


    /**
     * Left multiplies a Householder matrix {@code H=I-}&alpha{@code vv}<sup>T</sup>, represented by the vector
     * {@code v}, to another matrix {@code A}. That is, computes {@code H}<sup>T</sup>{@code *A = (I-}&alpha{@code vv}<sup>T</sup
     * >{@code )}<sup>T</sup>{@code *A}.
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector {@code v}.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of sub-matrix in {@code src} to apply reflector to.
     * @param startRow Starting row of sub-matrix in {@code src} to apply reflector to.
     * @param endRow Starting row of sub-matrix in {@code src} to apply reflector to.
     * @param workArray An array to store temporary column data. This can help both with cache performance and reducing unneeded
     *                  garbage collection if this method is called repeatedly.
     */
    public static void leftMultReflector(Matrix src,
                                         double[] householderVector,
                                         double alpha,
                                         int startCol,
                                         int startRow, int endRow,
                                         double[] workArray) {
        // Note: this computes A - (alpha*v)*(v^T*A) rather than A - (alpha*v*v^T)*A.
        // The first method takes ~2(n^2 + n) flops while the second method takes ~(n^3 + n^2 + 2n) flops.

        int numCols = src.numCols;
        int srcRowOffset = startRow*numCols;
        double v0 = householderVector[startRow];

        for(int i=startCol; i<numCols; i++) {
            workArray[i] = v0*src.entries[srcRowOffset + i];
        }

        for(int k=startRow + 1; k<endRow; k++) {
            int srcIdx = k*numCols + startCol;
            double reflectorValue = householderVector[k];
            for(int i=startCol; i<numCols; i++) {
                workArray[i] += reflectorValue*src.entries[srcIdx++];
            }
        }

        RealOperations.scalMult(workArray, workArray, alpha, startCol, numCols);

        for(int i=startRow; i<endRow; i++) {
            double reflectorValue = householderVector[i];
            int indexA = i*numCols + startCol;

            for(int j=startCol; j<numCols; j++) {
                src.entries[indexA++] -= reflectorValue*workArray[j];
            }
        }
    }


    /**
     * Right multiplies a Householder matrix {@code H=I-}&alpha{@code vv}<sup>T</sup>, represented by the vector
     * {@code v}, to another matrix {@code A}. That is, computes {@code A*H = A*(I-}&alpha{@code vv}<sup>T</sup>{@code )}.
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector {@code v}.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of sub-matrix in {@code src} to apply reflector to.
     * @param startRow Starting row of sub-matrix in {@code src} to apply reflector to.
     * @param endRow Starting row of sub-matrix in {@code src} to apply reflector to.
     */
    public static void rightMultReflector(Matrix src,
                                          double[] householderVector,
                                          double alpha,
                                          int startCol,
                                          int startRow, int endRow) {

        for(int i=startCol; i<src.numRows; i++) {
            int startIndex = i*src.numCols + startRow;
            double sum = 0;
            int rowIndex = startIndex;

            for(int j = startRow; j < endRow; j++) {
                sum += src.entries[rowIndex++]*householderVector[j];
            }
            sum *= -alpha;

            rowIndex = startIndex;
            for(int j=startRow; j<endRow; j++) {
                src.entries[rowIndex++] += sum*householderVector[j];
            }
        }
    }


    /**
     * Left multiplies a Householder matrix {@code H=I-}&alpha{@code vv}<sup>H</sup>, represented by the vector
     * {@code v}, to another matrix {@code A}. That is, computes {@code H*A = (I-}&alpha{@code vv}<sup>H</sup>{@code )*A}.
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector {@code v}.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of sub-matrix in {@code src} to apply reflector to.
     * @param startRow Starting row of sub-matrix in {@code src} to apply reflector to.
     * @param endRow Starting row of sub-matrix in {@code src} to apply reflector to.
     * @param workArray An array to store temporary column data. This can help both with cache performance and reducing unneeded
     *                  garbage collection if this method is called repeatedly.
     */
    public static void leftMultReflector(CMatrix src,
                                         CNumber[] householderVector,
                                         CNumber alpha,
                                         int startCol,
                                         int startRow, int endRow,
                                         CNumber[] workArray) {
        int numCols = src.numCols;
        int srcRowOffset = startRow*numCols;
        CNumber v0 = householderVector[startRow].conj();

        for(int i=startCol; i<numCols; i++) {
            workArray[i] = v0.mult(src.entries[srcRowOffset + i]);
        }

        for(int k=startRow + 1; k<endRow; k++) {
            int srcIdx = k*numCols + startCol;
            CNumber reflectorValue = householderVector[k].conj();
            for(int i=startCol; i<numCols; i++) {
                workArray[i].addEq(reflectorValue.mult(src.entries[srcIdx++]));
            }
        }

        ComplexOperations.scalMult(workArray, workArray, alpha, startCol, numCols);

        for(int i=startRow; i<endRow; i++) {
            CNumber reflectorValue = householderVector[i];
            int indexA = i*numCols + startCol;

            for(int j=startCol; j<numCols; j++) {
                src.entries[indexA++].subEq(reflectorValue.mult(workArray[j]));
            }
        }
    }


    /**
     * Right multiplies a Householder matrix {@code H=I-}&alpha{@code vv}<sup>H</sup>, represented by the vector
     * {@code v}, to another matrix {@code A}. That is, computes {@code A*H = A*(I-}&alpha{@code vv}<sup>H</sup>{@code )}.
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector {@code v}.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of sub-matrix in {@code src} to apply reflector to.
     * @param startRow Starting row of sub-matrix in {@code src} to apply reflector to.
     * @param endRow Starting row of sub-matrix in {@code src} to apply reflector to.
     */
    public static void rightMultReflector(CMatrix src,
                                          CNumber[] householderVector,
                                          CNumber alpha,
                                          int startCol,
                                          int startRow, int endRow) {
        CNumber negAlpha = alpha.addInv();

        for(int i=startCol; i<src.numRows; i++) {
            int startIndex = i*src.numCols + startRow;
            CNumber sum = CNumber.zero();
            int rowIndex = startIndex;

            for(int j = startRow; j < endRow; j++) {
                sum.addEq(src.entries[rowIndex++].mult(householderVector[j]));
            }
            sum.multEq(negAlpha);

            rowIndex = startIndex;
            for(int j=startRow; j<endRow; j++) {
                src.entries[rowIndex++].addEq(sum.mult(householderVector[j].conj()));
            }
        }
    }


    /**
     * Applies a Householder matrix {@code H=I-}&alpha{@code vv}<sup>T</sup>, represented by the vector {@code v} to a
     * matrix {@code A}. That is, computes {@code H*A*H = }
     */
    public static void symmLeftRightMultReflector(Matrix src,
                                                  double[] u,
                                                  double gamma,
                                                  int row,
                                                  double[] w) {
        int N = src.numRows;

        // compute v = -gamma*A*u
        for(int i=row; i<N; i++) {
            double total = 0;
            // the lower triangle is not written to so it needs to traverse upwards
            // to get the information. Reduces the number of matrix writes need
            // improving large matrix performance
            for(int j=row; j<i; j++) {
                total += src.entries[j*N + i]*u[j];
                System.out.printf("accessing: (%d, %d)\n", j, i);
            }
            for(int j=i; j<N; j++) {
                total += src.entries[i*N + j]*u[j];
                System.out.printf("accessing: (%d, %d)\n", i, j);
            }

            w[i] = -gamma*total;
        }

        // alpha = -0.5*gamma*u^T*v
        double alpha = 0;

        for(int i=row; i<N; i++) {
            alpha += u[i]*w[i];
        }
        alpha *= -0.5*gamma;

        // w = v + alpha*u
        for(int i=row; i<N; i++) {
            w[i] += alpha*u[i];
        }

        // A = A + w*u^T + u*w^T
        for(int i=row; i<N; i++) {

            double ww = w[i];
            double uu = u[i];

            int rowA = i*N;
            for(int j=i; j<N; j++) {
                // only write to the upper portion of the matrix
                // this reduces the number of cache misses
                src.entries[rowA + j] += ww*u[j] + w[j]*uu;
            }
        }
    }


    /**
     * Left multiplies a Householder matrix {@code H=I-}&alpha{@code vv}<sup>T</sup>, represented by the vector
     * {@code v}, to another matrix {@code A}. That is, computes {@code H*A = (I-}&alpha{@code vv}<sup>T</sup>{@code )*A}.
     *
     * <p>This method is significantly more efficient than forming the full Householder matrix and multiplying it to the other
     * matrix.</p>
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector {@code v}.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of sub-matrix in {@code src} to apply reflector to.
     * @param startRow Starting row of sub-matrix in {@code src} to apply reflector to.
     * @param endRow Starting row of sub-matrix in {@code src} to apply reflector to.
     */
    public static void leftMultReflector(Matrix src,
                                         Vector householderVector,
                                         double alpha,
                                         int startCol,
                                         int startRow, int endRow) {
        leftMultReflector(src, householderVector.entries, alpha, startCol, startRow, endRow, new double[src.numCols-startCol]);
    }


    /**
     * Right multiplies a Householder matrix {@code H=I-}&alpha{@code vv}<sup>T</sup>, represented by the vector
     * {@code v}, to another matrix {@code A}. That is, computes {@code A*H = A*(I-}&alpha{@code vv}<sup>T</sup>{@code )}.
     *
     * <p>This method is significantly more efficient than forming the full Householder matrix and multiplying it to the other
     * matrix.</p>
     *
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector {@code v}.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of sub-matrix in {@code src} to apply reflector to.
     * @param startRow Starting row of sub-matrix in {@code src} to apply reflector to.
     * @param endRow Starting row of sub-matrix in {@code src} to apply reflector to.
     */
    public static void rightMultReflector(Matrix src,
                                          Vector householderVector,
                                          double alpha,
                                          int startCol,
                                          int startRow, int endRow) {

        rightMultReflector(src, householderVector.entries, alpha, startCol, startRow, endRow);
    }


    /**
     * Left multiplies a Householder matrix {@code H=I-}&alpha{@code vv}<sup>H</sup>, represented by the vector
     * {@code v}, to another matrix {@code A}. That is, computes {@code H*A = (I-}&alpha{@code vv}<sup>H</sup>{@code )*A}.
     *
     * <p>This method is significantly more efficient than forming the full Householder matrix and multiplying it to the other
     * matrix.</p>
     *
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector {@code v}.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of sub-matrix in {@code src} to apply reflector to.
     * @param startRow Starting row of sub-matrix in {@code src} to apply reflector to.
     * @param endRow Starting row of sub-matrix in {@code src} to apply reflector to.
     */
    public static void leftMultReflector(CMatrix src,
                                         CVector householderVector,
                                         CNumber alpha,
                                         int startCol,
                                         int startRow, int endRow) {
        leftMultReflector(src, householderVector.entries, alpha, startCol, startRow, endRow, new CNumber[src.numCols-startCol]);
    }


    /**
     * Right multiplies a Householder matrix {@code H=I-}&alpha{@code vv}<sup>H</sup>, represented by the vector
     * {@code v}, to another matrix {@code A}. That is, computes {@code A*H = A*(I-}&alpha{@code vv}<sup>H</sup>{@code )}.
     *
     * <p>This method is significantly more efficient than forming the full Householder matrix and multiplying it to the other
     * matrix.</p>
     *
     * @param src Source matrix apply Householder vector to (modified).
     * @param householderVector Householder vector {@code v}.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of sub-matrix in {@code src} to apply reflector to.
     * @param startRow Starting row of sub-matrix in {@code src} to apply reflector to.
     * @param endRow Starting row of sub-matrix in {@code src} to apply reflector to.
     */
    public static void rightMultReflector(CMatrix src,
                                          CVector householderVector,
                                          CNumber alpha,
                                          int startCol,
                                          int startRow, int endRow) {
        rightMultReflector(src, householderVector.entries, alpha, startCol, startRow, endRow);
    }
}
