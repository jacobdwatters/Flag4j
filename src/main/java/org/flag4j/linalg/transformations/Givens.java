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

package org.flag4j.linalg.transformations;


import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

/**
 * <p>This class contains methods for computing real or complex Givens' rotation matrices.
 * 
 * <p>A Givens' rotator is a square matrix G(i, k, theta) which, when left multiplied to a vector, represents
 * a counterclockwise rotation of theta radians of the vector in the (i, j) plane. Givens rotators
 * are a unitary transformation.
 */
public final class Givens {

    private Givens() {
        // Hide default constructor for utility class.
        
    }


    /**
     * Constructs a general Givens rotation matrix. This method can be numerically unstable. See
     * {@link #getRotator(Vector, int)} if the goal is to zero an element of a vector using a Givens rotator. This method
     * will produce more accurate results in general and is more robust to overflows.
     * @param size The size of the Givens' rotation matrix.
     * @param i Index of the first axis to rotate through.
     * @param j Index of the second axis to rotate through.
     * @param theta Angle in radians of rotation through the (i, j) plane.
     * @return A Givens' rotation matrix with specified {@code size} which, when left multiplied to a vector,
     * represents a counterclockwise rotation of {@code theta} radians of the vector in the (i, j) plane.
     * @throws IndexOutOfBoundsException If {@code i} or {@code j} is greater than or equal to {@code size}.
     */
    public static Matrix getGeneralRotator(int size, int i, int j, double theta) {
        ValidateParameters.ensureIndicesInBounds(size, i, j);
        if(i==j) throw new IllegalArgumentException("The indices i and j cannot be equal.");

        // Initialize rotator as identity matrix.
        Matrix G = Matrix.I(size);

        double c = Math.cos(theta);
        double s = Math.sin(theta);

        G.data[i*(G.numCols + 1)] = c;
        G.data[i*G.numCols + j] = s;
        G.data[j*G.numCols + i] = -s;
        G.data[j*(G.numCols + 1)] = c;

        return G;
    }


    /**
     * Constructs a Givens rotator G such that for a vector v,
     * Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>] where r<sub>i</sub>=0.
     * That is, when the rotator G is left multiplied to the vector v, it zeros out the entry at position {@code i}.
     * @param v Vector to construct Givens rotator for.
     * @param i Position to zero out when applying the rotator to {@code v}.
     * @return A Givens rotator G such that for a vector v,
     * Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>] where r<sub>i</sub>=0.
     * @throws IndexOutOfBoundsException If {@code i} is not in the range [0, v.size).
     */
    public static Matrix getRotator(Vector v, int i) {
        return getRotator(v.data, i);
    }


    /**
     * Constructs a Givens rotator G such that for a vector v,
     * Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>] where r<sub>i</sub>=0.
     * That is, when the rotator G is left multiplied to the vector v,
     * it zeros out the entry at position {@code i}.
     * @param v Vector to construct Givens rotator for.
     * @param i Position to zero out when applying the rotator to {@code v}.
     * @return A Givens rotator G such that for a vector v,
     * Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>] where r<sub>i</sub>=0.
     * @throws IndexOutOfBoundsException If {@code i} is not in the range [0, v.size).
     */
    public static Matrix getRotator(double[] v, int i) {
        ValidateParameters.ensureIndicesInBounds(v.length, i);

        double[] cs = stableTrigVals(v[0], v[i]);

        Matrix G = Matrix.I(v.length); // Initialize rotator to identity matrix.

        G.data[i*(G.numCols + 1)] = cs[0];
        G.data[i*G.numCols] = -cs[1];
        G.data[i] = cs[1];
        G.data[0] = cs[0];

        return G;
    }


    /**
     * Constructs a Givens rotator G such that for a vector v,
     * Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>] where r<sub>i</sub>=0.
     * That is, when the rotator G is left multiplied to the vector v,
     * it zeros out the entry at position {@code i}.
     * @param v Vector to construct Givens rotator for.
     * @param i Position to zero out when applying the rotator to {@code v}.
     * @return A Givens rotator G such that for a vector v,
     * Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>] where r<sub>i</sub>=0.
     * @throws IndexOutOfBoundsException If {@code i} is not in the range [0, v.size).
     */
    public static CMatrix getRotator(CVector v, int i) {
        ValidateParameters.ensureIndicesInBounds(v.size, i);

        double r = VectorNorms.norm(v.data);
        Complex128 c = v.data[0].div(r);
        Complex128 s = v.data[i].div(r);

        CMatrix G = CMatrix.I(v.size); // Initialize rotator to identity matrix.

        G.data[i*(G.numCols + 1)] = c.conj();
        G.data[i*G.numCols] = s.addInv();
        G.data[i] = s;
        G.data[0] = c;

        return G;
    }


    /**
     * Constructs a Givens rotator G of size 2 such that for a vector v = [a, b] we have
     * Gv = [r, 0].
     * @param v Vector of size 2 to construct Givens rotator for.
     * @return A Givens rotator G of size 2 such that for a vector v = [a, b] we have
     * Gx = [r, 0].
     * @throws IllegalArgumentException If the vector {@code v} is not of size 2.
     */
    public static Matrix get2x2Rotator(Vector v) {
        ValidateParameters.ensureArrayLengthsEq(2, v.size);
        return get2x2Rotator(v.data[0], v.data[1]);
    }


    /**
     * Constructs a Givens rotator G of size 2 such that for a vector v = [a, b] nwe have Gv = [r, 0].
     * @param v0 First entry in vector to construct rotator for.
     * @param v1 Second entry in vector to construct rotator for.
     * @return A Givens rotator G of size 2 such that for a vector v = [a, b] we have
     * Gx = [r, 0].
     * @throws IllegalArgumentException If the vector {@code v} is not of size 2.
     */
    public static Matrix get2x2Rotator(double v0, double v1) {
        double[] cs = stableTrigVals(v0, v1);

        return new Matrix(new double[][]{
                {cs[0], cs[1]},
                {-cs[1], cs[0]}
        });
    }


    /**
     * <p>Left multiplies a 2x2 Givens rotator to a matrix at the specified row. This is done in place.
     *
     * <p>Specifically, computes G*A[i-1:i+1][i-1:i+1] where i={@code row}, G is the 2x2 Givens rotator,
     * A is the matrix to apply the reflector to, and A[i-1:i+1][i-1:]
     * represents the slice of A the reflector effects which has shape (2, A.numCols - i - 1).
     *
     * <p>This method is likely to be faster than computing this multiplication explicitly.
     *
     * @param src The matrix to left multiply the rotator to (modified).
     * @param G The 2x2 givens rotator. Note, the size is not explicitly checked.
     * @param row The row to the rotator is being applied to.
     * @param workArray Array to store temporary values. If null, a new array will be created (modified).
     * @throws ArrayIndexOutOfBoundsException If the {@code workArray} is not at least large enough to store the
     * {@code 2*(A.numCols - i - 1)} data.
     */
    public static void leftMult2x2Rotator(Matrix src, Matrix G, int row, double[] workArray) {
        double[] src1 = G.data;
        double[] src2 = src.data;

        int cols2 = src.numCols;
        int destCols = (cols2 - (row-1));
        if(workArray==null) workArray = new double[2*destCols];

        int m = row-1;
        int src2Row1 = m*cols2 + m;
        int src2Row2 = row*cols2 + m;

        double g11 = src1[0];
        double g12 = src1[1];
        double g21 = src1[2];
        double g22 = src1[3];

        for(int j=m; j<cols2; j++) {
            int destIdx1 = j - m;
            int destIdx2 = destCols + destIdx1;

            workArray[destIdx1] += g11*src2[src2Row1];
            workArray[destIdx2] += g21*src2[src2Row1++];
            workArray[destIdx1] += g12*src2[src2Row2];
            workArray[destIdx2] += g22*src2[src2Row2++];
        }

        System.arraycopy(workArray, 0, src2, m*cols2 + m, destCols);
        System.arraycopy(workArray, destCols, src2, row*cols2 + m, destCols);
    }


    /**
     * <p>Right multiplies a 2x2 Givens rotator to a matrix at the specified row. This is done in place
     *
     * <p>Specifically, computes A[:][i-1:i+1]*G<sup>H</sup>
     * where i={@code row}, G is the 2x2 Givens rotator, A is the matrix to apply the reflector to, and A[:i+1][i-1:i+1]
     * represents the slice of A the reflector effects which has shape {(row+1, 2).
     *
     * <p>This method is likely to be faster than computing this multiplication explicitly.
     *
     * @param src The matrix to left multiply the rotator to (modified).
     * @param G The 2x2 givens rotator. Note, the size is not explicitly checked.
     * @param row The row to the rotator is being applied to.
     * @param workArray Array to store temporary values. If null, a new array will be created (modified).
     * If the {@code workArray} is not at least large enough to store the
     * {@code 2*(row+1)} data.
     */
    public static void rightMult2x2Rotator(Matrix src, Matrix G, int row, double[] workArray) {
        double[] src1 = src.data;
        double[] src2 = G.data;

        int cols1 = src.numCols;
        int rows1 = src.numRows;
        if(workArray==null) workArray = new double[2*rows1]; // Has shape (row+1, 2)

        int m = row - 1;

        double g11 = src2[0];
        double g12 = src2[1];
        double g21 = src2[2];
        double g22 = src2[3];

        // Apply the rotator.
        for(int i=0; i<rows1; i++) {
            int tempIdx1 = i*2;
            int tempIdx2 = tempIdx1 + 1;
            int src1Idx1 = i*cols1 + m;
            int src1Idx2 = src1Idx1 + 1;

            workArray[tempIdx1] += src1[src1Idx1]*g11;
            workArray[tempIdx1] += src1[src1Idx2]*g12;
            workArray[tempIdx2] += src1[src1Idx1]*g21;
            workArray[tempIdx2] += src1[src1Idx2]*g22;
        }

        // Copy result back into source matrix.
        for(int i=0; i<rows1; i++) {
            src1[i*cols1 + m] = workArray[i*2];
            src1[i*cols1 + row] = workArray[i*2 + 1];
        }
    }


    /**
     * <p>Left multiplies a 2x2 Givens rotator to a matrix at the specified row. This is done in place.
     *
     * <p>Specifically, computes G*A[i-1:i+1][i-1:i+1] where i={@code row},  G is the 2x2 Givens rotator,
     * A is the matrix to apply the reflector to, and {A[i-1:i+1][i-1:]
     * represents the slice of A the reflector effects which has shape (2, A.numCols - i - 1).
     *
     * <p>This method is likely to be faster than computing this multiplication explicitly.
     *
     * @param src The matrix to left multiply the rotator to (modified).
     * @param G The 2x2 givens rotator. Note, the size is not explicitly checked.
     * @param row The row to the rotator is being applied to.
     * @param workArray Array to store temporary values. If null, a new array will be created (modified).
     * If the {@code workArray} is not at least large enough to store the
     * {@code 2*(A.numCols - i - 1)} data.
     */
    public static void leftMult2x2Rotator(CMatrix src, CMatrix G, int row, Complex128[] workArray) {
        Complex128[] src1 = G.data;
        Complex128[] src2 = src.data;

        int cols2 = src.shape.get(1);
        int destCols = (cols2 - (row-1));
        if(workArray==null) workArray = new Complex128[2*destCols];
        Arrays.fill(workArray, 0, 2*destCols, Complex128.ZERO);

        int m = row-1;
        int src2Row1 = m*cols2 + m;
        int src2Row2 = row*cols2 + m;

        Complex128 g11 = src1[0];
        Complex128 g12 = src1[1];
        Complex128 g21 = src1[2];
        Complex128 g22 = src1[3];

        // Apply the rotator.
        for(int j=m; j<cols2; j++) {
            int destIdx1 = j - m;
            int destIdx2 = destCols + destIdx1;

            workArray[destIdx1] = workArray[destIdx1].add(g11.mult((Complex128) src2[src2Row1]));
            workArray[destIdx2] = workArray[destIdx2].add(g21.mult((Complex128) src2[src2Row1++]));
            workArray[destIdx1] = workArray[destIdx1].add(g12.mult((Complex128) src2[src2Row2]));
            workArray[destIdx2] = workArray[destIdx2].add(g22.mult((Complex128) src2[src2Row2++]));
        }

        // Copy result back into src matrix.
        System.arraycopy(workArray, 0, src2, m*cols2 + m, destCols);
        System.arraycopy(workArray, destCols, src2, row*cols2 + m, destCols);
    }


    /**
     * <p>Right multiplies a 2x2 Givens rotator to a matrix at the specified row. This is done in place
     *
     * <p>Specifically, computes A[:][i-1:i+1]*G<sup>H</sup>
     * where i={@code row}, G is the 2x2 Givens rotator, A is the matrix to apply the reflector to, and
     * {@code A[:i+1][i-1:i+1]} represents the slice of A the reflector effects which has shape {@code (row+1, 2)}.
     *
     * <p>This method is likely to be faster than computing this multiplication explicitly.
     *
     * @param src The matrix to left multiply the rotator to (modified).
     * @param G The 2x2 givens rotator. Note, the size is not explicitly checked.
     * @param row The row to the rotator is being applied to.
     * @param workArray Array to store temporary values. If null, a new array will be created (modified).
     * If the {@code workArray} is not at least large enough to store the 
     * {@code 2*(row+1)} data.
     */
    public static void rightMult2x2Rotator(CMatrix src, CMatrix G, int row, Complex128[] workArray) {
        Complex128[] src1 = src.data;
        Complex128[] src2 = G.data;

        int cols1 = src.numCols;
        int rows1 = src.numRows;
        if(workArray==null) workArray = new Complex128[2*rows1]; // Has shape (row+1, 2)
        Arrays.fill(workArray, Complex128.ZERO);

        int m = row - 1;

        Complex128 g11 = src2[0].conj();
        Complex128 g12 = src2[1].conj();
        Complex128 g21 = src2[2].conj();
        Complex128 g22 = src2[3].conj();

        // Apply the rotator.
        for(int i=0; i<rows1; i++) {
            int tempIdx1 = i*2;
            int tempIdx2 = tempIdx1 + 1;
            int src1Idx1 = i*cols1 + m;
            int src1Idx2 = src1Idx1 + 1;

            workArray[tempIdx1] = workArray[tempIdx1].add(src1[src1Idx1].mult(g11));
            workArray[tempIdx1] = workArray[tempIdx1].add(src1[src1Idx2].mult(g12));
            workArray[tempIdx2] = workArray[tempIdx2].add(src1[src1Idx1].mult(g21));
            workArray[tempIdx2] = workArray[tempIdx2].add(src1[src1Idx2].mult(g22));
        }

        // Copy result back into source matrix.
        for(int i=0; i<rows1; i++) {
            src1[i*cols1 + m] = workArray[i*2];
            src1[i*cols1 + row] = workArray[i*2 + 1];
        }
    }


    /**
     * Constructs a Givens rotator G of size 2 such that for a vector v = [a, b] we have Gv = [r, 0].
     * @param v Vector to construct Givens rotator for.
     * @return A Givens rotator G of size 2 such that for a vector v = [a, b] we have Gx = [r, 0].
     * @throws IllegalArgumentException If {@code v.size != 2}.
     */
    public static CMatrix get2x2Rotator(CVector v) {
        ValidateParameters.ensureArrayLengthsEq(2, v.size);

        return get2x2Rotator((Complex128) v.data[0], (Complex128) v.data[1]);
    }


    /**
     * Constructs a Givens rotator G of size 2 such that for a vector v = [a, b] we have Gv = [r, 0].
     * @param v0 First entry in vector to construct rotator for.
     * @param v1 Second entry in vector to construct rotator for.
     * @return A Givens rotator {@code G} of size 2 such that for a vector v = [a, b] we have Gx = [r, 0].
     * @throws IllegalArgumentException If {@code v.size != 2}.
     */
    public static CMatrix get2x2Rotator(Complex128 v0, Complex128 v1) {
        double maxAbs = Math.max(v0.abs(), v1.abs());

        Complex128 v0Scale = v0.div(maxAbs);
        Complex128 v1Scale = v1.div(maxAbs);

        double r = Math.sqrt(v0Scale.magSquared() + v1Scale.magSquared())*maxAbs;
        Complex128 c = v0.div(r);
        Complex128 s = v1.div(r);

        return new CMatrix(new Complex128[][]{
                {c.conj(), s},
                {s.addInv(), c}
        });
    }


    /**
     * Computes the sine and cosine values for a Givens' rotation in a stable manner which reduces the risk of over/underflow problems.
     * @return An array of length two containing in order the cosine value and sine value for the Givens' rotation.
     */
    private static double[] stableTrigVals(double a, double b) {
        // Stable computation of sine/cosine which is more robust to overflow issues.
        double t;
        double s;
        double c;

        if(b==0) {
            c = a==0 ? 1.0 : Math.signum(a);
            s = 0;

        } else if(a==0) {
            c = 0;
            s = Math.signum(b);

        } else if(Math.abs(a) > Math.abs(b)) {
            t = a / b;
            s = Math.signum(b) / Math.sqrt(1.0 + t*t);
            c = s*t;

        } else {
            t = b / a;
            c = Math.signum(a) / Math.sqrt(1.0 + t*t);
            s = c*t;
        }

        return new double[]{c, s};
    }
}
