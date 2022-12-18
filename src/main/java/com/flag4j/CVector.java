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

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.VectorBase;
import com.flag4j.core.VectorOrientations;
import com.flag4j.util.ArrayUtils;

/**
 * Complex dense vector. Vectors may be oriented as row vectors, column vectors, or unoriented.
 * See {@link VectorOrientations} for orientations.
 */
public class CVector extends VectorBase<CNumber[]> {

    /**
     * Creates a column vector of specified size filled with zeros.
     * @param size Size of the vector.
     */
    public CVector(int size) {
        super(size, VectorOrientations.COL, new CNumber[size]);
        ArrayUtils.fillZeros(super.entries);
    }


    /**
     * Creates a column vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public CVector(int size, double fillValue) {
        super(size, VectorOrientations.COL, new CNumber[size]);
        ArrayUtils.fill(super.entries, fillValue);
    }


    /**
     * Creates a column vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public CVector(int size, CNumber fillValue) {
        super(size, VectorOrientations.COL, new CNumber[size]);
        ArrayUtils.fill(super.entries, fillValue);
    }


    /**
     * Creates a vector of specified size filled with zeros.
     * @param size Size of the vector.
     * @param orientation Orientation of the vector.
     */
    public CVector(int size, VectorOrientations orientation) {
        super(size, orientation, new CNumber[size]);
        ArrayUtils.fillZeros(super.entries);
    }


    /**
     * Creates a vector of specified size filled with zeros.
     * @param size Size of the vector.
     * @param fillValue Fills array with
     * @param orientation Orientation of the vector.
     */
    public CVector(int size, double fillValue, VectorOrientations orientation) {
        super(size, orientation, new CNumber[size]);
        ArrayUtils.fill(super.entries, fillValue);
    }


    /**
     * Creates a vector of specified size filled with zeros.
     * @param size Size of the vector.
     * @param fillValue Fills array with
     * @param orientation Orientation of the vector.
     */
    public CVector(int size, CNumber fillValue, VectorOrientations orientation) {
        super(size, orientation, new CNumber[size]);
        ArrayUtils.fill(super.entries, fillValue);
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public CVector(double[] entries) {
        super(entries.length, VectorOrientations.COL, new CNumber[entries.length]);
        ArrayUtils.copy2CNumber(entries, super.entries);
    }

    /**
     * Creates a vector with specified entries and orientation.
     * @param entries Entries for this column vector.
     * @param orientation Orientation of the vector.
     */
    public CVector(double[] entries, VectorOrientations orientation) {
        super(entries.length, orientation, new CNumber[entries.length]);
        ArrayUtils.copy2CNumber(entries, super.entries);
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public CVector(int[] entries) {
        super(entries.length, VectorOrientations.COL, new CNumber[entries.length]);
        ArrayUtils.copy2CNumber(entries, super.entries);
    }

    /**
     * Creates a vector with specified entries and orientation.
     * @param entries Entries for this column vector.
     * @param orientation Orientation of the vector.
     */
    public CVector(int[] entries, VectorOrientations orientation) {
        super(entries.length, orientation, new CNumber[entries.length]);
        ArrayUtils.copy2CNumber(entries, super.entries);
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public CVector(CNumber[] entries) {
        super(entries.length, VectorOrientations.COL, entries);
    }


    /**
     * Creates a vector with specified entries and orientation.
     * @param entries Entries for this column vector.
     * @param orientation Orientation of the vector.
     */
    public CVector(CNumber[] entries, VectorOrientations orientation) {
        super(entries.length, orientation, entries);
    }


    /**
     * Constructs a complex vector whose entries and shape are specified by another complex vector.
     * @param a Complex vector to copy.
     */
    public CVector(CVector a) {
        super(a.size(), a.getOrientation(), new CNumber[a.totalEntries().intValue()]);
        ArrayUtils.copy2CNumber(a.entries, super.entries);
    }
}
