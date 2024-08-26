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

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

import java.util.Arrays;


/**
 * This class contains methods for computing real or complex Givens' rotation matrices.
 * A Givens' rotator is a square matrix {@code G(i, k, theta)} which, when left multiplied to a vector, represents
 * a counterclockwise rotation of {@code theta} radians of the vector in the {@code (i, j)} plane. Givens rotators
 * are a unitary transformation.
 */
public final class Givens {

    private Givens() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Constructs a general Givens rotation matrix. This method can be numerically unstable. See
     * {@link #getRotator(VectorOld, int)} if the goal is to zero an element of a vector using a Givens rotator. This method
     * will produce more accurate results in general and is more robust to overflows.
     * @param size The size of the Givens' rotation matrix.
     * @param i Index of the first axis to rotate through.
     * @param j Index of the second axis to rotate through.
     * @param theta Angle in radians of rotation through the {@code (i, j)} plane.
     * @return A Givens' rotation matrix with specified {@code size} which, when left multiplied to a vector,
     * represents a counterclockwise rotation of {@code theta} radians of the vector in the {@code (i, j)} plane.
     * @throws IndexOutOfBoundsException If {@code i} or {@code j} is greater than or equal to {@code size}.
     */
    public static MatrixOld getGeneralRotator(int size, int i, int j, double theta) {
        ParameterChecks.ensureIndexInBounds(size, i, j);
        if(i==j) throw new IllegalArgumentException("The indices i and j cannot be equal.");

        // Initialize rotator as identity matrix.
        MatrixOld G = MatrixOld.I(size);

        double c = Math.cos(theta);
        double s = Math.sin(theta);

        G.entries[i*(G.numCols + 1)] = c;
        G.entries[i*G.numCols + j] = s;
        G.entries[j*G.numCols + i] = -s;
        G.entries[j*(G.numCols + 1)] = c;

        return G;
    }


    /**
     * Constructs a Givens rotator {@code G} such that for a vector {@code v},
     * {@code Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>]} where r<sub>i</sub>=0.
     * That is, when the rotator {@code G} is left multiplied to the vector {@code v},
     * it zeros out the entry at position {@code i}.
     * @param v VectorOld to construct Givens rotator for.
     * @param i Position to zero out when applying the rotator to v.
     * @return A Givens rotator {@code G} such that for a vector {@code v},
     * {@code Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>]} where r<sub>i</sub>=0.
     * @throws IndexOutOfBoundsException If {@code i} is not in the range {@code [0, v.size)}.
     */
    public static MatrixOld getRotator(VectorOld v, int i) {
        return getRotator(v.entries, i);
    }


    /**
     * Constructs a Givens rotator {@code G} such that for a vector {@code v},
     * {@code Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>]} where r<sub>i</sub>=0.
     * That is, when the rotator {@code G} is left multiplied to the vector {@code v},
     * it zeros out the entry at position {@code i}.
     * @param v VectorOld to construct Givens rotator for.
     * @param i Position to zero out when applying the rotator to v.
     * @return A Givens rotator {@code G} such that for a vector {@code v},
     * {@code Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>]} where r<sub>i</sub>=0.
     * @throws IndexOutOfBoundsException If {@code i} is not in the range {@code [0, v.size)}.
     */
    public static MatrixOld getRotator(double[] v, int i) {
        ParameterChecks.ensureIndexInBounds(v.length, i);

        double[] cs = stableTrigVals(v[0], v[i]);

        MatrixOld G = MatrixOld.I(v.length); // Initialize rotator to identity matrix.

        G.entries[i*(G.numCols + 1)] = cs[0];
        G.entries[i*G.numCols] = -cs[1];
        G.entries[i] = cs[1];
        G.entries[0] = cs[0];

        return G;
    }


    /**
     * Constructs a Givens rotator {@code G} such that for a vector {@code v},
     * {@code Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>]} where r<sub>i</sub>=0.
     * That is, when the rotator {@code G} is left multiplied to the vector {@code v},
     * it zeros out the entry at position {@code i}.
     * @param v VectorOld to construct Givens rotator for.
     * @param i Position to zero out when applying the rotator to v.
     * @return A Givens rotator {@code G} such that for a vector {@code v},
     * {@code Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>]} where r<sub>i</sub>=0.
     * @throws IndexOutOfBoundsException If {@code i} is not in the range {@code [0, v.size)}.
     */
    public static CMatrixOld getRotator(CVectorOld v, int i) {
        ParameterChecks.ensureIndexInBounds(v.size, i);

        double r = VectorNorms.norm(v);
        CNumber c = v.entries[0].div(r);
        CNumber s = v.entries[i].div(r);

        CMatrixOld G = CMatrixOld.I(v.size); // Initialize rotator to identity matrix.

        G.entries[i*(G.numCols + 1)] = c.conj();
        G.entries[i*G.numCols] = s.addInv();
        G.entries[i] = s;
        G.entries[0] = c;

        return G;
    }


    /**
     * Constructs a Givens rotator {@code G} of size 2 such that for a vector {@code v = [a, b]} we have
     * {@code Gv = [r, 0]}.
     * @param v VectorOld of size 2 to construct Givens rotator for.
     * @return A Givens rotator {@code G} of size 2 such that for a vector {@code v = [a, b]} we have
     * {@code Gx = [r, 0]}.
     * @throws IllegalArgumentException If the vector {@code v} is not of size 2.
     */
    public static MatrixOld get2x2Rotator(VectorOld v) {
        ParameterChecks.ensureArrayLengthsEq(2, v.size);
        return get2x2Rotator(v.entries[0], v.entries[1]);
    }


    /**
     * Constructs a Givens rotator {@code G} of size 2 such that for a vector {@code v = [a, b]} we have
     * {@code Gv = [r, 0]}.
     * @param v0 First entry in vector to construct rotator for.
     * @param v1 Second entry in vector to construct rotator for.
     * @return A Givens rotator {@code G} of size 2 such that for a vector {@code v = [a, b]} we have
     * {@code Gx = [r, 0]}.
     * @throws IllegalArgumentException If the vector {@code v} is not of size 2.
     */
    public static MatrixOld get2x2Rotator(double v0, double v1) {
        double[] cs = stableTrigVals(v0, v1);

        return new MatrixOld(new double[][]{
                {cs[0], cs[1]},
                {-cs[1], cs[0]}
        });
    }


    /**
     * <p>Left multiplies a 2x2 Givens rotator to a matrix at the specified row. This is done in place.</p>
     *
     * <p>Specifically, computes {@code G*A[i-1:i+1][i-1:i+1]} where {@code i=row}, {@code G} is the 2x2 Givens rotator,
     * {@code A} is the matrix to apply the reflector to, and {@code A[i-1:i+1][i-1:]}
     * represents the slice of {@code A} the reflector effects which has shape {@code (2, A.numCols - i - 1)}.</p>
     *
     * <p>This method is likely to be faster than computing this multiplication explicitly.</p>
     *
     * @param src The matrix to left multiply the rotator to (modified).
     * @param G The 2x2 givens rotator. Note, the size is not explicitly checked.
     * @param row The row to the rotator is being applied to.
     * @param workArray Array to store temporary values. If null, a new array will be created (modified).
     * @throws ArrayIndexOutOfBoundsException If the {@code workArray} is not at least large enough to store the
     * {@code 2*(A.numCols - i - 1)} entries.
     */
    public static void leftMult2x2Rotator(MatrixOld src, MatrixOld G, int row, double[] workArray) {
        double[] src1 = G.entries;
        double[] src2 = src.entries;

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
     * <p>Right multiplies a 2x2 Givens rotator to a matrix at the specified row. This is done in place</p>
     *
     * <p>Specifically, computes {@code A[:][i-1:i+1]*G}<sup>H</sup>
     * where {@code i=row}, {@code G} is the 2x2 Givens rotator, {@code A} is the matrix to apply the reflector to, and {@code A[:i+1][i-1:i+1]}
     * represents the slice of {@code A} the reflector effects which has shape {@code (row+1, 2)}.</p>
     *
     * <p>This method is likely to be faster than computing this multiplication explicitly.</p>
     *
     * @param src The matrix to left multiply the rotator to (modified).
     * @param G The 2x2 givens rotator. Note, the size is not explicitly checked.
     * @param row The row to the rotator is being applied to.
     * @param workArray Array to store temporary values. If null, a new array will be created (modified).
     * If the {@code workArray} is not at least large enough to store the
     * {@code 2*(row+1)} entries.
     */
    public static void rightMult2x2Rotator(MatrixOld src, MatrixOld G, int row, double[] workArray) {
        double[] src1 = src.entries;
        double[] src2 = G.entries;

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
     * <p>Left multiplies a 2x2 Givens rotator to a matrix at the specified row. This is done in place.</p>
     *
     * <p>Specifically, computes {@code G*A[i-1:i+1][i-1:i+1]} where {@code i=row}, {@code G} is the 2x2 Givens rotator,
     * {@code A} is the matrix to apply the reflector to, and {@code A[i-1:i+1][i-1:]}
     * represents the slice of {@code A} the reflector effects which has shape {@code (2, A.numCols - i - 1)}.</p>
     *
     * <p>This method is likely to be faster than computing this multiplication explicitly.</p>
     *
     * @param src The matrix to left multiply the rotator to (modified).
     * @param G The 2x2 givens rotator. Note, the size is not explicitly checked.
     * @param row The row to the rotator is being applied to.
     * @param workArray Array to store temporary values. If null, a new array will be created (modified).
     * If the {@code workArray} is not at least large enough to store the
     * {@code 2*(A.numCols - i - 1)} entries.
     */
    public static void leftMult2x2Rotator(CMatrixOld src, CMatrixOld G, int row, CNumber[] workArray) {
        CNumber[] src1 = G.entries;
        CNumber[] src2 = src.entries;

        int cols2 = src.shape.get(1);
        int destCols = (cols2 - (row-1));
        if(workArray==null) workArray = new CNumber[2*destCols];
        Arrays.fill(workArray, 0, 2*destCols, CNumber.ZERO);

        int m = row-1;
        int src2Row1 = m*cols2 + m;
        int src2Row2 = row*cols2 + m;

        CNumber g11 = src1[0];
        CNumber g12 = src1[1];
        CNumber g21 = src1[2];
        CNumber g22 = src1[3];

        // Apply the rotator.
        for(int j=m; j<cols2; j++) {
            int destIdx1 = j - m;
            int destIdx2 = destCols + destIdx1;

            workArray[destIdx1] = workArray[destIdx1].add(g11.mult(src2[src2Row1]));
            workArray[destIdx2] = workArray[destIdx2].add(g21.mult(src2[src2Row1++]));
            workArray[destIdx1] = workArray[destIdx1].add(g12.mult(src2[src2Row2]));
            workArray[destIdx2] = workArray[destIdx2].add(g22.mult(src2[src2Row2++]));
        }

        // Copy result back into src matrix.
        System.arraycopy(workArray, 0, src2, m*cols2 + m, destCols);
        System.arraycopy(workArray, destCols, src2, row*cols2 + m, destCols);
    }


    /**
     * <p>Right multiplies a 2x2 Givens rotator to a matrix at the specified row. This is done in place</p>
     *
     * <p>Specifically, computes {@code A[:][i-1:i+1]*G}<sup>H</sup>
     * where {@code i=row}, {@code G} is the 2x2 Givens rotator, {@code A} is the matrix to apply the reflector to, and {@code A[:i
     * +1][i
     * -1:i+1]}
     * represents the slice of {@code A} the reflector effects which has shape {@code (row+1, 2)}.</p>
     *
     * <p>This method is likely to be faster than computing this multiplication explicitly.</p>
     *
     * @param src The matrix to left multiply the rotator to (modified).
     * @param G The 2x2 givens rotator. Note, the size is not explicitly checked.
     * @param row The row to the rotator is being applied to.
     * @param workArray Array to store temporary values. If null, a new array will be created (modified).
     * If the {@code workArray} is not at least large enough to store the 
     * {@code 2*(row+1)} entries.     
     */
    public static void rightMult2x2Rotator(CMatrixOld src, CMatrixOld G, int row, CNumber[] workArray) {
        CNumber[] src1 = src.entries;
        CNumber[] src2 = G.entries;

        int cols1 = src.numCols;
        int rows1 = src.numRows;
        if(workArray==null) workArray = new CNumber[2*rows1]; // Has shape (row+1, 2)
        Arrays.fill(workArray, CNumber.ZERO);

        int m = row - 1;

        CNumber g11 = src2[0].conj();
        CNumber g12 = src2[1].conj();
        CNumber g21 = src2[2].conj();
        CNumber g22 = src2[3].conj();

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
     * Constructs a Givens rotator {@code G} of size 2 such that for a vector {@code v = [a, b]} we have
     * {@code Gv = [r, 0]}.
     * @param v VectorOld to construct Givens rotator for.
     * @return A Givens rotator {@code G} of size 2 such that for a vector {@code v = [a, b]} we have
     * {@code Gx = [r, 0]}.
     * @throws IllegalArgumentException If the vector {@code v} is not of size 2.
     */
    public static CMatrixOld get2x2Rotator(CVectorOld v) {
        ParameterChecks.ensureArrayLengthsEq(2, v.size);

        return get2x2Rotator(v.entries[0], v.entries[1]);
    }


    /**
     * Constructs a Givens rotator {@code G} of size 2 such that for a vector {@code v = [a, b]} we have
     * {@code Gv = [r, 0]}.
     * @param v0 First entry in vector to construct rotator for.
     * @param v1 Second entry in vector to construct rotator for.
     * @return A Givens rotator {@code G} of size 2 such that for a vector {@code v = [a, b]} we have
     * {@code Gx = [r, 0]}.
     * @throws IllegalArgumentException If the vector {@code v} is not of size 2.
     */
    public static CMatrixOld get2x2Rotator(CNumber v0, CNumber v1) {
        double maxAbs = Math.max(v0.abs(), v1.abs());

        CNumber v0Scale = v0.div(maxAbs);
        CNumber v1Scale = v1.div(maxAbs);

        double r = Math.sqrt(v0Scale.magSquared() + v1Scale.magSquared())*maxAbs;
        CNumber c = v0.div(r);
        CNumber s = v1.div(r);

        return new CMatrixOld(new CNumber[][]{
                {c.conj(), s},
                {s.addInv(), c}
        });
    }


    /**
     * Computes the sine and cosine values for a Givens' rotation in a stable manner which avoids
     * any possibility of overflow.
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
