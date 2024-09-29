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

package org.flag4j;


import org.flag4j.arrays.Shape;

abstract class Field<T extends Field<T>> {
    abstract T add(T b);
    abstract T sub(T b);
    abstract T mult(T b);
    abstract T div(T b);
}

class Complex extends Field<Complex> {
    final double re;
    final double im;

    Complex(final double re, final double im) {
        this.re = re;
        this.im = im;
    }

    Complex add(Complex b) {
        return new Complex(re + b.re, im + b.im);
    }

    Complex sub(Complex b) {
        return new Complex(re - b.re, im - b.im);
    }

    Complex mult(Complex b) {
        return new Complex(this.re*b.re - this.im*b.im, this.re*b.im + this.im*b.re);
    }

    Complex div(Complex b) {
        double divisor = b.re*b.re + b.im*b.im;
        return new Complex((re*b.re + im*b.im) / divisor, (im*b.re - re*b.im) / divisor);
    }
}

abstract class TestFieldMatrix<T extends TestFieldMatrix<T, U>, U extends Field<U>> {
    Shape shape;
    Field<U>[] entries;

    TestFieldMatrix(Shape shape, Field<U>[] entries) {
        this.shape = shape;
        this.entries = entries;
    }

    abstract T makeLikeMatrix(Shape shape, Field<U>[] entries);

    T add(T b) {
        Field<U>[] sum = new Field[entries.length];

        for(int i=0; i<entries.length; i++)
            sum[i] = entries[i].add((U) b.entries[i]);

        return makeLikeMatrix(shape, sum);
    }
}

class ComplexMatrix extends TestFieldMatrix<ComplexMatrix, Complex> {
    ComplexMatrix(Shape shape, Field<Complex>[] entries) {
        super(shape, entries);
    }

    @Override
    ComplexMatrix makeLikeMatrix(Shape shape, Field<Complex>[] entries) {
        return new ComplexMatrix(shape, entries);
    }
}

public class Test {
    public static void main(String[] args) {
        Complex[] entries = {new Complex(1, 3), new Complex(2, 4), new Complex(5, 6), new Complex(7, 8)};
        Shape shape = new Shape(2, 2);
        ComplexMatrix matrix = new ComplexMatrix(shape, entries);

        ComplexMatrix sum = matrix.add(matrix);
    }
}
