package com.flag4j.linalg.transformations;

import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.Matrix;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;


/**
 * This class contains methods for computing real or complex Givens' rotation matrices.
 * A Givens' rotator is a square matrix {@code G(i, k, theta)} which, when left multiplied to a vector, represents
 * a counterclockwise rotation of {@code theta} radians of the vector in the {@code (i, j)} plane. Givens rotators
 * represent a unitary transformation.
 */
public class Givens {

    private Givens() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Constructs a general Givens rotation matrix. This method can be numerically unstable. See
     * {@link #getRotator(Vector, int)} if the goal is to zero an element of a vector using a Givens rotator. This method
     * will produce more accurate results in general and is more robust to overflows.
     * @param size The size of the Givens' rotation matrix.
     * @param i Index of the first axis to rotate through.
     * @param j Index of the second axis to rotate through.
     * @param theta Angle in radians of rotation through the {@code (i, j)} plane.
     * @return A Givens' rotation matrix with specified {@code size} which, when left multiplied to a vector,
     * represents a counterclockwise rotation of {@code theta} radians of the vector in the {@code (i, j)} plane.
     * @throws IndexOutOfBoundsException If {@code i} or {@code j} is greater than or equal to {@code size}.
     */
    public static Matrix getGeneralRotator(int size, int i, int j, double theta) {
        ParameterChecks.assertIndexInBounds(size, i, j);

        // Initialize rotator as identity matrix.
        Matrix G = Matrix.I(size);

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
     * @param v Vector to construct Givens rotator for.
     * @param i Position to zero out when applying the rotator to v.
     * @return A Givens rotator {@code G} such that for a vector {@code v},
     * {@code Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>]} where r<sub>i</sub>=0.
     * @throws IndexOutOfBoundsException If {@code i} is not in the range {@code [0, v.size)}.
     */
    public static Matrix getRotator(Vector v, int i) {
        ParameterChecks.assertIndexInBounds(v.size, i);

        double[] cs = stableTrigVals(v.entries[0], v.entries[i]);

        Matrix G = Matrix.I(v.size); // Initialize rotator to identity matrix.

        G.entries[i*(G.numCols + 1)] = cs[0];
        G.entries[i*G.numCols] = cs[1];
        G.entries[i] = -cs[1];
        G.entries[0] = cs[0];

        return G;
    }


    /**
     * Constructs a Givens rotator {@code G} such that for a vector {@code v},
     * {@code Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>]} where r<sub>i</sub>=0.
     * That is, when the rotator {@code G} is left multiplied to the vector {@code v},
     * it zeros out the entry at position {@code i}.
     * @param v Vector to construct Givens rotator for.
     * @param i Position to zero out when applying the rotator to v.
     * @return A Givens rotator {@code G} such that for a vector {@code v},
     * {@code Gv = [r<sub>1</sub> ... r<sub>i</sub> ... r<sub>n</sub>]} where r<sub>i</sub>=0.
     */
    public static CMatrix getRotator(CVector v, int i) {
        double r = v.norm();
        CNumber c = v.entries[0].div(r);
        CNumber s = v.entries[1].div(r);

        CMatrix G = CMatrix.I(v.size); // Initialize rotator to identity matrix.

        G.entries[i*(G.numCols + 1)] = c.conj();
        G.entries[i*G.numCols] = s;
        G.entries[i] = s.addInv();
        G.entries[0] = c;

        return G;
    }


    /**
     * Constructs a Givens rotator {@code G} of size 2 such that for a vector {@code v = [a, b]} we have
     * {@code Gv = [r, 0]}.
     * @param v Vector of size 2 to construct Givens rotator for.
     * @return A Givens rotator {@code G} of size 2 such that for a vector {@code v = [a, b]} we have
     * {@code Gx = [r, 0]}.
     * @throws IllegalArgumentException If the vector {@code v} is not of size 2.
     */
    public static Matrix get2x2Rotator(Vector v) {
        ParameterChecks.assertArrayLengthsEq(2, v.size);

        double[] cs = stableTrigVals(v.entries[0], v.entries[1]);

        return new Matrix(new double[][]{
                {cs[0], cs[1]},
                {-cs[1], cs[0]}
        });
    }


    /**
     * Constructs a Givens rotator {@code G} of size 2 such that for a vector {@code v = [a, b]} we have
     * {@code Gv = [r, 0]}.
     * @param v Vector to construct Givens rotator for.
     * @return A Givens rotator {@code G} of size 2 such that for a vector {@code v = [a, b]} we have
     * {@code Gx = [r, 0]}.
     * @throws IllegalArgumentException If the vector {@code v} is not of size 2.
     */
    public static CMatrix get2x2Rotator(CVector v) {
        double r = v.norm();
        CNumber c = v.entries[0].div(r);
        CNumber s = v.entries[1].div(r);

        return new CMatrix(new CNumber[][]{
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
        // Stable computation of sine/cosine which avoids any possible overflow.
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
            double bSgn = b==0 ? 1 : Math.signum(b);
            s = bSgn / Math.sqrt(1 + t*t);
            c = s*t;

        } else {
            t = b / a;
            double aSgn = a==0 ? 1 : Math.signum(a);
            c = aSgn / Math.sqrt(1 + t*t);
            s = c*t;
        }

        return new double[]{c, s};
    }
}
