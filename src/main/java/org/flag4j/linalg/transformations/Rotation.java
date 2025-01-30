/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;


/**
 * <p>The {@code Rotation} class provides utility methods to compute rotation matrices for 2D and 3D rotations.
 * These matrices are designed to perform rotations of vectors in a counterclockwise direction when viewed from a
 * positive axis perspective in a right-handed coordinate system.
 *
 * <p>
 *
 * <p>This class supports:
 * <ul>
 *   <li>2D rotation matrices for rotating column vectors by a specified angle in degrees.</li>
 *   <li>3D rotation matrices for rotating about the x-axis, y-axis, and z-axis.</li>
 *   <li>3D rotation matrices for yaw-pitch-roll rotations.</li>
 *   <li>3D rotation matrices for arbitrary axis rotations.</li>
 *   <li>3D rotation matrices for proper Euler angle rotations.</li>
 * </ul>
 *
 * <p>Rotation matrices have the following properties:
 * <ul>
 *   <li>A rotation matrix is orthogonal: <b></b>R<sup>-1</sup></b> = <b>R</b><sup>T</sup>.</li>
 *   <li>Rotations preserve the length of vectors (orthogonal transformations).</li>
 *   <li>The inverse/transpose rotation matrix undoes the rotation: <b>x</b> = <b>R</b><sup>T</sup><b>Rx</b> =
 *   <b>RR</b><sup>T</sup><b>x</b></li>
 *   <li>Successive rotations can be composed through matrix multiplication (rotation order is from right to left).</li>
 * </ul>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 *         // Rotate a 2D vector by 45 degrees.
 *         double theta = 45.0;
 *         Matrix rotation2D = Rotation.rotate2D(theta);
 *         Vector vector2D = new Vector(1, 0);  // A vector along the x-axis
 *         Vector rotatedVector2D = rotation2D.mult(vector2D);
 *
 *         // Rotate a 3D vector about the x-axis by 90 degrees.
 *         double thetaX = 90.0;
 *         Matrix rotationX3D = Rotation.rotateX3D(thetaX);
 *         Vector vector3D = new Vector(0, 1, 0);  // A vector along the y-axis
 *         Vector rotatedVector3D = rotationX3D.mult(vector3D);
 *
 *         // Perform a yaw-pitch-roll rotation in 3D.
 *         double yaw = 30.0;
 *         double pitch = 45.0;
 *         double roll = 60.0;
 *         Matrix yawPitchRoll = Rotation.rotate3D(yaw, pitch, roll);
 *         vector3D = new Vector(1, 1, 1);  // An arbitrary 3D vector
 *         Vector rotatedVector = yawPitchRoll.mult(vector3D);
 *
 *         // Rotate a 3D vector about an arbitrary axis.
 *         Vector axis = new Vector(1, 1, 0);  // An arbitrary axis.
 *         double angle = 45.0;
 *         Matrix arbitraryAxisRotation = Rotation.rotate3D(angle, axis);
 *         Vector arbitraryRotatedVector = arbitraryAxisRotation.mult(vector3D);
 *
 *         // Perform a rotation using proper Euler angles.
 *         double alpha = 30.0;  // Rotation about z-axis
 *         double beta = 45.0;   // Rotation about x-axis
 *         double gamma = 60.0;  // Rotation about z-axis again
 *         Matrix eulerRotation = Rotation.rotateEuler3D(alpha, beta, gamma);
 *         Vector eulerRotatedVector = eulerRotation.mult(vector3D);
 *
 *         // Perform multiple rotations.
 *         double thetaY = 90.0;
 *         double thetaZ = -30.0;
 *         Matrix rotationY3D = Rotation.rotateY3D(thetaY);
 *         Matrix rotationZ3D = Rotation.rotateZ3D(thetaZ);
 *         vector3D = new Vector(0, 1, 0);
 *         // First rotate by thetaY then by thetaZ.
 *         Matrix combinedRotation3D = rotationZ3D.mult(rotationY3D);
 *         Vector multiRotatedVector3D = combinedRotation3D.mult(vector3D);
 * }</pre>
 *
 * <p><strong>Note:</strong> Methods involving sequential rotations, such as yaw-pitch-roll or Euler angles, are
 * susceptible to gimbal lock, where two axes align, causing a loss of one degree of rotational freedom. For
 * applications requiring continuous rotations, consider alternative representations such as quaternions.
 */
public final class Rotation {

    private Rotation() {
        // Hide default constructor for utility class.
    }

    // TODO: Consider having an object-oriented class for rotations like in scipy.spatial.transform.Rotation


    /**
     * <p>Constructs a rotation matrix, <b>R</b>(&theta;), which rotates 2D column vectors {@code theta} degrees.
     * When {@code theta > 0} the rotation is counterclockwise. When
     *
     * <p>A 2D rotation matrix <b>R</b>(&theta;), rotates a 2D column vector <b>x</b>, &theta; degrees by means of
     * the following matrix-vector multiplication:
     * <pre>
     *     <b>x</b>' = <b>R</b>(&theta;)<b>x</b></pre>
     * The following holds <b>R</b>(-&theta;) = <b>R</b>(&theta;)<sup>-1</sup> = <b>R</b>(&theta;)<sup>T</sup>.
     * This means the inverse/transpose may be used to undo a rotation,
     * <pre>
     *     <b>x</b> = <b>R</b>(&theta;)<b>R</b>(&theta;)<sup>T</sup><b>x</b>
     *       = <b>R</b>(&theta;)<sup>T</sup><b>R</b>(&theta;)<b>x</b>
     *       = I<b>x</b></pre>
     *
     * @param theta The degrees to rotate a 2D vector by.
     * @return A rotation matrix which rotates (counterclockwise) 2D column vectors {@code theta} degrees.
     */
    public static Matrix rotate2D(double theta) {
        if(theta == 0) return Matrix.I(2);

        double rad = Math.toRadians(theta);
        double c = Math.cos(rad);
        double s = Math.sin(rad);

        return new Matrix(2, 2, c, -s, s, c);
    }


    /**
     * <p>Constructs a matrix which rotates 3D column vectors about the x-axis {@code theta} degrees. The rotation appears
     * counterclockwise when the x-axis points toward the observer, {@code theta > 0} and the coordinate system is right-handed.
     *
     * <p>A 3D rotation matrix, <b>R</b><sub>X</sub>(&theta;), rotates a 3D column vector <b>x</b> about the x-axis &theta; degrees by
     * means of the following matrix-vector multiplication:
     * <pre>
     *     <b>x</b>' = <b>R</b><sub>X</sub>(&theta;)<b>x</b></pre>
     * The following holds <b>R</b><sub>X</sub>(-&theta;) = <b>R</b><sub>X</sub>(&theta;)<sup>-1</sup> = <b>R</b><sub>X</sub>(&theta;)<sup>T</sup>.
     * This means the inverse/transpose may be used to undo a rotation,
     * <pre>
     *     <b>x</b> = <b>R</b><sub>X</sub>(&theta;)<b>R</b><sub>X</sub>(&theta;)<sup>T</sup><b>x</b>
     *       = <b>R</b><sub>X</sub>(&theta;)<sup>T</sup><b>R</b><sub>X</sub>(&theta;)<b>x</b>
     *       = I<b>x</b></pre>
     *
     *
     * @param theta The degrees to rotate a 3D vector about the x-axis by.
     * @return matrix which rotates 3D column vectors about the x-axis {@code theta} degrees.
     */
    public static Matrix rotateX3D(double theta) {
        if(theta == 0) return Matrix.I(3);

        double rad = Math.toRadians(theta);
        double c = Math.cos(rad);
        double s = Math.sin(rad);

        return new Matrix(3, 3,
                1, 0, 0,
                0, c, -s,
                0, s, c
        );
    }


    /**
     * <p>Constructs a matrix which rotates 3D column vectors about the y-axis {@code theta} degrees. The rotation appears
     * counterclockwise when the y-axis points toward the observer, {@code theta > 0} and the coordinate system is right-handed.
     *
     * <p>A 3D rotation matrix, <b>R</b><sub>Y</sub>(&theta;), rotates a 3D column vector <b>x</b> about the y-axis &theta; degrees by
     * means of the following matrix-vector multiplication:
     * <pre>
     *     <b>x</b>' = <b>R</b><sub>Y</sub>(&theta;)<b>x</b></pre>
     * The following holds <b>R</b><sub>Y</sub>(-&theta;) = <b>R</b><sub>Y</sub>(&theta;)<sup>-1</sup> = <b>R</b><sub>Y</sub>(&theta;)<sup>T</sup>.
     * This means the inverse/transpose may be used to undo a rotation,
     * <pre>
     *     <b>x</b> = <b>R</b><sub>Y</sub>(&theta;)<b>R</b><sub>Y</sub>(&theta;)<sup>T</sup><b>x</b>
     *       = <b>R</b><sub>Y</sub>(&theta;)<sup>T</sup><b>R</b><sub>Y</sub>(&theta;)<b>x</b>
     *       = I<b>x</b></pre>
     *
     *
     * @param theta The degrees to rotate a 3D vector about the y-axis by.
     * @return matrix which rotates 3D column vectors about the y-axis {@code theta} degrees.
     */
    public static Matrix rotateY3D(double theta) {
        if(theta == 0) return Matrix.I(3);

        double rad = Math.toRadians(theta);
        double c = Math.cos(rad);
        double s = Math.sin(rad);

        return new Matrix(3, 3,
                c, 0, s,
                0, 1, 0,
                -s, 0, c
        );
    }


    /**
     * <p>Constructs a matrix which rotates 3D column vectors about the z-axis {@code theta} degrees. The rotation appears
     * counterclockwise when the z-axis points toward the observer, {@code theta > 0} and the coordinate system is right-handed.
     *
     * <p>A 3D rotation matrix, <b>R</b><sub>Z</sub>(&theta;), rotates a 3D column vector <b>x</b> about the z-axis &theta; degrees by
     * means of the following matrix-vector multiplication:
     * <pre>
     *     <b>x</b>' = <b>R</b><sub>Z</sub>(&theta;)<b>x</b></pre>
     * The following holds <b>R</b><sub>Z</sub>(-&theta;) = <b>R</b><sub>Z</sub>(&theta;)<sup>-1</sup> = <b>R</b><sub>Z</sub>(&theta;)<sup>T</sup>.
     * This means the inverse/transpose may be used to undo a rotation,
     * <pre>
     *     <b>x</b> = <b>R</b><sub>Z</sub>(&theta;)<b>R</b><sub>Z</sub>(&theta;)<sup>T</sup><b>x</b>
     *       = <b>R</b><sub>Z</sub>(&theta;)<sup>T</sup><b>R</b><sub>Z</sub>(&theta;)<b>x</b>
     *       = I<b>x</b></pre>
     *
     *
     * @param theta The degrees to rotate a 3D vector about the z-axis by.
     * @return matrix which rotates 3D column vectors about the z-axis {@code theta} degrees.
     */
    public static Matrix rotateZ3D(double theta) {
        if(theta == 0) return Matrix.I(3);

        double rad = Math.toRadians(theta);
        double c = Math.cos(rad);
        double s = Math.sin(rad);

        return new Matrix(3, 3,
                c, -s, 0,
                s, c, 0,
                0, 0, 1
        );
    }


    /**
     * <p>Constructs a 3D rotation matrix, <b>R</b>(&alpha;, &beta;, &gamma;), representing a rotation with yaw, pitch, and roll
     * angles
     * &alpha;, &beta;, and &gamma; respectively. This is equivalent to rotating by &alpha; degrees about the x-axis,
     * &beta; degrees about the y-axis, and &gamma; degrees about the z-axis in that order. Each of the three rotations appear
     * counterclockwise when the axis about which they occur points toward the observer,
     * the rotation angle is positive, and the coordinate system is right-handed.
     *
     * <p>A 3D rotation matrix, <b>R</b>(&alpha;, &beta;, &gamma;), rotates a 3D column vector <b>x</b>,
     * &gamma;, &beta;, and &alpha; degrees about the x-, y-, and z-axes in that order by means of the following matrix
     * multiplication:
     * <pre>
     *     <b>x</b>' = <b>R</b>(&alpha;, &beta;, &gamma;)<b>x</b>
     *        = <b>R</b><sub>Z</sub>(&gamma;)<b>R</b><sub>Y</sub>(&beta;)<b>R</b><sub>X</sub>(&alpha;)x</pre>
     *
     * <p><strong>Note:</strong> This method is susceptible to gimbal lock, a phenomenon where two of the rotation axes align,
     * causing a loss of one degree of rotational freedom. Gimbal lock occurs when the second rotation in the sequence aligns
     * the axes, such as when the pitch angle (for yaw-pitch-roll) or the second Euler angle (for proper Euler angles) is
     * &plusmn;90&deg;. To avoid gimbal lock, consider using rotation representations that do not rely on sequential rotations.
     *
     * @param yaw Degrees to rotate about the vertical (yaw) axis (i.e. the z-axis).
     * @param pitch Degrees to rotate about the lateral (pitch) axis (i.e. the y-axis).
     * @param roll Degrees to rotate about the longitudinal (roll) axis (i.e. the x-axis).
     * @return a rotation matrix representing a rotation with yaw, pitch, and roll angles &alpha;, &beta;, and &gamma; respectively.
     */
    public static Matrix rotate3D(double yaw, double pitch, double roll) {
        if(yaw == 0.0 && pitch == 0.0 && roll == 0.0) return Matrix.I(3);

        double radYaw = Math.toRadians(yaw);
        double radPitch = Math.toRadians(pitch);
        double radRoll = Math.toRadians(roll);

        double ca = Math.cos(radYaw);
        double sa = Math.sin(radYaw);
        double cb = Math.cos(radPitch);
        double sb = Math.sin(radPitch);
        double cy = Math.cos(radRoll);
        double sy = Math.sin(radRoll);

        return new Matrix(3, 3,
                ca*cb,  ca*sb*sy - sa*cy,   ca*sb*cy + sa*sy,
                sa*cb,  sa*sb*sy + ca*cy,   sa*sb*cy - ca*sy,
                -sb,    cb*sy,              cb*cy
        );
    }


    /**
     * <p>Constructs a 3D rotation matrix, <b>R<sub>u</sub></b>(&theta;), which representing a rotation of &theta; degrees about
     * an axis unit vector <b>u</b>. The rotation is a counterclockwise rotation when <b>u</b> points towards the observer, &theta;> 0,
     * and the coordinate system is right-handed.
     *
     * <p>A 3D rotation matrix, <b>R<sub>u</sub></b>(&theta;), rotates a 3D column vector <b>x</b>,
     * &theta; degrees about the vector <b>u</b> by means of the following matrix multiplication:
     * <pre>
     *     <b>x</b>' = <b>R<sub>u</sub></b>(&theta;)<b>x</b></pre>
     *
     * @param theta The degrees to rotate about the vector <b>u</b>.
     * @param axis The axis vector <b>u</b> to rotate about. This vector will be normalized so it need not be a unit vector.
     * Must satisfy {@code axis.size == 3}.
     * @return
     */
    public static Matrix rotate3D(double theta, Vector axis) {
        if(axis.size != 3)
            throw new IllegalArgumentException("Axis vector size must be size 3 but got " + axis.size + ".");

        if(theta == 0.0) return Matrix.I(3);

        double rad = Math.toRadians(theta);
        double c = Math.cos(rad);
        double s = Math.sin(rad);

        double cInv = 1.0 - c;

        // Ensure vector is a unit vector.
        axis = axis.normalize();
        double ux = axis.data[0];
        double uy = axis.data[1];
        double uz = axis.data[2];

        // Construct using Rodrigues' formula: R = I + sin(t)*K + (1 - cos(t))*K^2 where
        //  where K is skew-symmetric with zeros on the diagonal:
        //      [  0  -uz  uy ]
        //  K = [  uz  0  -ux ]
        //      [ -uy  ux  0  ]
        double r01 = ux*uy*cInv;
        double r02 = ux*uz*cInv;
        double r12 = uy*uz*cInv;

        return new Matrix(3, 3,
                ux*ux*cInv + c, r01 - uz*s,     r02 + uy*s,
                r01 + uz*s,     uy*uy*cInv + c, r12 - ux*s,
                r02 - uy*s,     r12 + ux*s,     uz*uz*cInv + c
        );
    }


    /**
     * <p>Constructs a 3D rotation matrix, <b>R</b><sub>E</sub>(&alpha;, &beta;, &gamma;), representing a rotation described
     * by proper Euler angles (&alpha;, &beta;, &gamma;). This is equivalent to performing a rotation about the z-axis by &alpha;
     * degrees, then about the x-axis by &beta; degrees, then about the z-axis again by &gamma; degrees.
     *
     * <p>A 3D rotation matrix, <b>R</b><sub>E</sub>(&alpha;, &beta;, &gamma;), rotates a 3D column vector <b>x</b>,
     * according to the Euler angles (&alpha;, &beta;, &gamma;) by means of the following matrix multiplication:
     * <pre>
     *     <b>x</b>' = <b>R</b><sub>E</sub>(&alpha;, &beta;, &gamma;)<b>x</b>
     *        = <b>R</b><sub>Z</sub>(&gamma;)<b>R</b><sub>X</sub>(&beta;)<b>R</b><sub>Z</sub>(&alpha;)x</pre>
     *
     * <p><strong>Note:</strong> This method is susceptible to gimbal lock, a phenomenon where two of the rotation axes align,
     * causing a loss of one degree of rotational freedom. Gimbal lock occurs when the second rotation in the sequence aligns
     * the axes, such as when the pitch angle (for yaw-pitch-roll) or the second Euler angle (for proper Euler angles) is
     * &plusmn;90&deg;. To avoid gimbal lock, consider using rotation representations that do not rely on sequential rotations.
     *
     * @param alpha Degrees of first rotation about the z-axis.
     * @param beta Degrees of second rotation about the x-axis.
     * @param gamma Degrees of third rotation about the z-axis.
     * @return Constructs a rotation matrix representing a rotation described by proper Euler angles (&alpha;, &beta;, &gamma;).
     */
    public static Matrix rotateEuler(double alpha, double beta, double gamma) {
        if(alpha == 0.0 && beta == 0.0 && gamma == 0.0) return Matrix.I(3);

        double radAlpha = Math.toRadians(alpha);
        double radBeta = Math.toRadians(beta);
        double radGamma = Math.toRadians(gamma);

        double ca = Math.cos(radAlpha);
        double sa = Math.sin(radAlpha);
        double cb = Math.cos(radBeta);
        double sb = Math.sin(radBeta);
        double cy = Math.cos(radGamma);
        double sy = Math.sin(radGamma);

        return new Matrix(3, 3,
                cy*ca - sy*sa*cb,   -sy*cb*ca - sa*cy,  sy*sb,
                sy*ca + sa*cy*cb,   cy*cb*ca - sy*sa,   -sb*cy,
                sb*sa,              sb*ca,              cb
        );
    }
}
