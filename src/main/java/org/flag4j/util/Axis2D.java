/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package org.flag4j.util;

/**
 * Simple enum class for two-dimensional axis. The row axis has ordinal 0 and the column axis has ordinal 1.
 */
public enum Axis2D {
    /**
     * Row of 2D tensor.
     */
    ROW,
    /**
     * Column of 2D tensor
     */
    COL;

    /**
     * Get the ordinal of the row axis in 2D.
     * @return The ordinal of the row axis in 2D.
     */
    public static int row() {
        return ROW.ordinal();
    }


    /**
     * Get the ordinal of the column axis in 2D.
     * @return The ordinal of the column axis in 2D.
     */
    public static int col() {
        return COL.ordinal();
    }


    /**
     * Gets an array of all axes ordinals.
     * @return An array of all axes ordinals.
     */
    public static int[] allAxes() {
        return new int[]{ROW.ordinal(), COL.ordinal()};
    }
}
