/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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

package com.flag4j.linalg.transformations;

import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.Matrix;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ErrorMessages;

/**
 * This class contains methods for computing real or complex Householder reflectors (also known as elementary reflector).
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
     * @param normal The vector normal to the plane the Householder reflector will reflect through\.
     * @return A transformation matrix which describes a reflection through a plane containing the origin with the
     * specified {@code normal} vector, i.e. a Householder reflector.
     */
    public static Matrix getReflector(Vector normal) {
        Matrix H = Matrix.I(normal.size);
        Vector v = normal.copy();

        double signedNorm = -Math.copySign(v.norm(), v.entries[0]);
        v = v.div(v.entries[0] - signedNorm);
        v.entries[0] = 1;

        // Create projection matrix
        Matrix P = v.outer(v).mult(2.0/v.inner(v));
        H.subEq(P);

        return H;
    }


    /**
     * Computes the Householder reflector which describes a reflection through a hyperplane containing the origin which
     * is normal to the specified {@code normal} vector.
     * @param normal The vector normal to the plane the Householder reflector will reflect through\.
     * @return A transformation matrix which describes a reflection through a plane containing the origin with the
     * specified {@code normal} vector, i.e. a Householder reflector.
     */
    public static CMatrix getReflector(CVector normal) {
        CMatrix H = CMatrix.I(normal.size);
        CVector v = normal.copy();

        // Compute signed norm using modified sgn function.
        CNumber signedNorm = v.entries[0].equals(CNumber.ZERO) ?
                new CNumber(-v.norm()) : CNumber.sgn(v.entries[0]).mult(-v.norm());

        v = v.div(v.entries[0].sub(signedNorm));
        v.entries[0] = new CNumber(1);

        // Create projection matrix
        CMatrix P = v.outer(v).mult(2.0/v.inner(v).re);
        H.subEq(P);

        return H;
    }
}
