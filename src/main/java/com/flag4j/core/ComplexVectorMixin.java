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

package com.flag4j.core;


import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.SparseCMatrix;
import com.flag4j.SparseCVector;
import com.flag4j.complex_numbers.CNumber;

/**
 * This interface specifies methods which any complex vector should implement.
 * @param <T> Vector type.
 * @param <Y> Real vector type.
 */
public interface ComplexVectorMixin<T, Y> extends
        ComplexTensorMixin<T, Y>,
        VectorComparisonsMixin<T, CVector, SparseCVector, CVector, Y, CNumber>,
        VectorManipulationsMixin<T, CVector, SparseCVector, CVector, Y, CNumber,
                CMatrix, CMatrix, SparseCMatrix, CMatrix>,
        VectorOperationsMixin<T, CVector, SparseCVector, CVector, Y, CNumber,
                CMatrix, CMatrix, SparseCMatrix, CMatrix>,
        VectorPropertiesMixin<T, CVector, SparseCVector, CVector, Y, CNumber>{
}
