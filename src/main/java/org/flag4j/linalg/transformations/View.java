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


import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

/**
 * Utility class for generating view matrices.
 */
public final class View {

    private View() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Constructs a view matrix for a camera at world position {@code cameraPos}, looking at the point {@code center} where {@code up}
     * is the direction of the upwards vector in world coordinates. This function is identical to glm's lookAt function.
     * @param cameraPos Camera position in world coordinates. Must have length 3.
     * @param center The point the camera is looking at. Must have length 3.
     * @param up The vector in the direction fo the worlds 'upward' vector. Must have length 3.
     * @return A view matrix for a camera at world position {@code cameraPos}, looking at the point {@code center} where {@code up}
     * is the direction of the upwards vector in world coordinates.
     * @throws IllegalArgumentException If any of the argument vectors do not have length 3.
     */
    public static MatrixOld lookAt(VectorOld cameraPos, VectorOld center, VectorOld up) {
        ParameterChecks.ensureEquals(3, cameraPos.size, center.size, up.size);

        VectorOld f = center.sub(cameraPos).normalize();
        VectorOld u = up.normalize();
        VectorOld s = f.cross(u).normalize();
        u = s.cross(f);

        MatrixOld view = new MatrixOld(4);

        view.entries[0] = s.entries[0];
        view.entries[4] = s.entries[1];
        view.entries[8] = s.entries[2];

        view.entries[1] = u.entries[0];
        view.entries[5] = u.entries[1];
        view.entries[9] = u.entries[2];

        view.entries[2] = -f.entries[0];
        view.entries[6] = -f.entries[1];
        view.entries[10] = -f.entries[2];

        view.entries[12] = -s.inner(cameraPos);
        view.entries[13] = -u.inner(cameraPos);
        view.entries[14] = f.inner(cameraPos);

        view.entries[15] = 1.0;

        return view;
    }
}
