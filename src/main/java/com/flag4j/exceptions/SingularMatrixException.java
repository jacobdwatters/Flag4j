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

package com.flag4j.exceptions;

/**
 * An exception which is thrown when some operation not defined for singular matrices is attempted to be
 * performed on a singular matrix. For example, attempting to invert a singular matrix.
 */
public class SingularMatrixException extends LinearAlgebraException {

    private static final String INFO = "Matrix is singular.";

    /**
     * Creates a SingularMatrixException with the simple error message "Matrix is singular."
     */
    public SingularMatrixException() {
        super(INFO);
    }

    /**
     * Creates a SingularMatrixException with a specified error message. Note, the string " Matrix is singular." will
     * be automatically appended to the error message.
     * @param errMsg Error message to display when this SingularMatrixException is thrown.
     */
    public SingularMatrixException(String errMsg) {
        super(errMsg + " " + INFO);
    }
}
