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

        double signedNorm = -Math.copySign(normal.norm(), normal.entries[0]);
        v = normal.div(normal.entries[0] - signedNorm);
        v.entries[0] = 1;

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
     * @param normal Vector normal to the plane which {@code H} reflects across.
     * @return The vector {@code v} in of a Householder matrix {@code H=I-2vv}<sup>T</sup> which reflects across a plane
     * normal to {@code normal}.
     */
    public static Vector getVector(Vector normal) {
        double normX = normal.norm();
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
     * @param normal The vector normal to the plane the Householder reflector will reflect through\.
     * @return A transformation matrix which describes a reflection through a plane containing the origin with the
     * specified {@code normal} vector, i.e. a Householder reflector.
     */
    public static CMatrix getReflector(CVector normal) {
        CVector v;

        // Compute signed norm using modified sgn function.
        CNumber signedNorm = normal.entries[0].equals(0) ?
                new CNumber(-normal.norm()) : CNumber.sgn(normal.entries[0]).mult(-normal.norm());

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
}
