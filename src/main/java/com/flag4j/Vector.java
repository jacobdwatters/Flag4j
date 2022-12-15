/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
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

package com.flag4j;

import com.flag4j.core.VectorBase;
import com.flag4j.core.VectorOrientations;

import java.util.Arrays;


/**
 * Real dense vector. Vectors may be oriented as row vectors, column vectors, or unoriented.
 * See {@link VectorOrientations} for orientations.
 */
public class Vector extends VectorBase<double[]> {


    /**
     * Creates a column vector of specified size filled with zeros.
     * @param size Size of the vector.
     */
    public Vector(int size) {
        super(size, VectorOrientations.COL, new double[size]);
    }


    /**
     * Creates a column vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public Vector(int size, double fillValue) {
        super(size, VectorOrientations.COL, new double[size]);
        Arrays.fill(super.entries, fillValue);
    }


    /**
     * Creates a vector of specified size filled with zeros.
     * @param size Size of the vector.
     * @param orientation Orientation of the vector.
     */
    public Vector(int size, VectorOrientations orientation) {
        super(size, orientation, new double[size]);
    }


    /**
     * Creates a vector of specified size filled with zeros.
     * @param size Size of the vector.
     * @param fillValue Fills array with
     * @param orientation Orientation of the vector.
     */
    public Vector(int size, double fillValue, VectorOrientations orientation) {
        super(size, orientation, new double[size]);
        Arrays.fill(super.entries, fillValue);
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public Vector(double[] entries) {
        super(entries.length, VectorOrientations.COL, entries.clone());
    }

    /**
     * Creates a vector with specified entries and orientation.
     * @param entries Entries for this column vector.
     * @param orientation Orientation of the vector.
     */
    public Vector(double[] entries, VectorOrientations orientation) {
        super(entries.length, orientation, entries.clone());
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public Vector(int[] entries) {
        super(entries.length, VectorOrientations.COL, new double[entries.length]);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = entries[i];
        }
    }

    /**
     * Creates a vector with specified entries and orientation.
     * @param entries Entries for this column vector.
     * @param orientation Orientation of the vector.
     */
    public Vector(int[] entries, VectorOrientations orientation) {
        super(entries.length, orientation, new double[entries.length] );

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = entries[i];
        }
    }


    /**
     * Creates a vector from another vector. This essentially copies the vector.
     * @param a Vector to make copy of.
     */
    public Vector(Vector a) {
        super(a.entries.length, a.getOrientation(), a.entries.clone());
    }
}
