/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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

package org.flag4j.arrays.backend.smart_visitors;

import org.flag4j.algebraic_structures.Field;
import org.flag4j.algebraic_structures.Ring;
import org.flag4j.algebraic_structures.Semiring;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.dense.*;
import org.flag4j.arrays.sparse.*;

// TODO: Docs
public abstract class MatrixVisitor<T> {

    protected final T other;

    public MatrixVisitor(T other) {
        this.other = other;
    }


    protected <U> String getInvalidOppMessage(MatrixMixin<?, ?, ?, ?> matrix) {
        return "Operation is not supported for matrix/vector types: "
                + matrix.getClass().getSimpleName() + " and " + other.getClass().getSimpleName();
    }


    public abstract T visit(Matrix matrix);
    public abstract T visit(CooMatrix matrix);
    public abstract T visit(CsrMatrix matrix);


    public abstract T visit(CMatrix matrix);
    public abstract T visit(CooCMatrix matrix);
    public abstract T visit(CsrCMatrix matrix);


    public abstract <U extends Field<U>> T visit(FieldMatrix<U> matrix);
    public abstract <U extends Field<U>> T visit(CooFieldMatrix<U> matrix);
    public abstract <U extends Field<U>> T visit(CsrFieldMatrix<U> matrix);


    public abstract <U extends Semiring<U>> T visit(SemiringMatrix<U> matrix);
    public abstract <U extends Semiring<U>> T visit(CooSemiringMatrix<U> matrix);
    public abstract <U extends Semiring<U>> T visit(CsrSemiringMatrix<U> matrix);


    public abstract <U extends Ring<U>> T visit(RingMatrix<U> matrix);
    public abstract <U extends Ring<U>> T visit(CooRingMatrix<U> matrix);
    public abstract <U extends Ring<U>> T visit(CsrRingMatrix<U> matrix);
}
