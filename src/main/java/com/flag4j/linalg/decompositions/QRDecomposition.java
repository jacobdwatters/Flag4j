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

package com.flag4j.linalg.decompositions;

import com.flag4j.core.MatrixMixin;

/**
 * <p>This abstract class specifies methods for computing the {@code QR} decomposition of a matrix.</p>
 * <p>The {@code QR} decomposition, decomposes a matrix {@code A} into a unitary matrix {@code Q}
 * and an upper triangular matrix {@code R} such that {@code A=QR}.</p>
 */
public abstract class QRDecomposition<T extends MatrixMixin<T, ?, ?, ?, ?, ?>> implements Decomposition<T> {

    /**
     * Storage matrix for {@code Q}.
     */
    protected T Q;
    /**
     * Storage matrix for {@code R}.
     */
    protected T R;
    /**
     * Flag for determining if the full {@code QR} decomposition should be used. If true, the full {@code QR} decomposition will be computed,
     * if false, the reduced QR decomposition will be computed.
     */
    protected final boolean fullQR;

    /**
     * Constructs {@code QR} decomposer which computes the full {@code QR} decomposition.
     */
    public QRDecomposition() {
        fullQR = true;
    }

    /**
     * Constructs a {@code QR} decomposer which computes either the full or reduced {@code QR} decomposition.
     * @param fullQR Flag for determining if the full {@code QR} decomposition should be used.
     *               If true, the full {@code QR} decomposition will be computed, if false,
     *               the reduced {@code QR} decomposition will be computed.
     */
    public QRDecomposition(boolean fullQR) {
        this.fullQR = fullQR;
    }


    /**
     * Applies {@code QR} decomposition to the source matrix.
     *
     * @param src The source matrix to decompose. Not modified.
     * @return A reference to this decomposer.
     */
    @Override
    public QRDecomposition<T> decompose(T src) {
        if(fullQR) {
            this.full(src);
        } else {
            this.reduced(src);
        }

        return this;
    }


    /**
     * Computes the reduced QR decomposition on the src matrix.
     * @param src The source matrix to decompose.
     */
    protected abstract void full(T src);


    /**
     * Computes the full QR decomposition on the src matrix.
     * @param src The source matrix to decompose.
     */
    protected abstract void reduced(T src);


    /**
     * Gets the {@code Q} matrix from the {@code QR} decomposition.
     * @return The {@code Q} matrix from the {@code QR} decomposition.
     */
    public T getQ(){
        return Q;
    }


    /**
     * Gets the {@code R} matrix from the {@code QR} decomposition.
     * @return The {@code R} matrix from the {@code QR} decomposition.
     */
    public T getR(){
        return R;
    }
}
