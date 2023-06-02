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

package com.flag4j.cvector;

import com.flag4j.CVector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CVectorZeroOneTests {

    CNumber[] aEntries;
    CVector a;

    @Test
    void zerosTest() {
        // ------------------ Sub-case 1 ------------------
        a = new CVector(34);
        assertTrue(a.isZeros());

        // ------------------ Sub-case 2 ------------------
        a = new CVector(0);
        assertTrue(a.isZeros());

        // ------------------ Sub-case 3 ------------------
        aEntries = new CNumber[]{new CNumber(), new CNumber(), new CNumber()};
        a = new CVector(aEntries);
        assertTrue(a.isZeros());

        // ------------------ Sub-case 4 ------------------
        aEntries = new CNumber[]{new CNumber(), new CNumber(1), new CNumber()};
        a = new CVector(aEntries);
        assertFalse(a.isZeros());

        // ------------------ Sub-case 5 ------------------
        aEntries = new CNumber[]{new CNumber(), new CNumber(9.4, -6.233), new CNumber()};
        a = new CVector(aEntries);
        assertFalse(a.isZeros());

        // ------------------ Sub-case 6 ------------------
        aEntries = new CNumber[]{new CNumber(), new CNumber(), new CNumber(0, -8.234)};
        a = new CVector(aEntries);
        assertFalse(a.isZeros());
    }


    @Test
    void onesTest() {
        // ------------------ Sub-case 1 ------------------
        a = new CVector(34, 1);
        assertTrue(a.isOnes());

        // ------------------ Sub-case 2 ------------------
        a = new CVector(0, 1);
        assertTrue(a.isOnes());

        // ------------------ Sub-case 3 ------------------
        aEntries = new CNumber[]{new CNumber(1), new CNumber(1), new CNumber(1)};
        a = new CVector(aEntries);
        assertTrue(a.isOnes());

        // ------------------ Sub-case 4 ------------------
        aEntries = new CNumber[]{new CNumber(1), new CNumber(1.2), new CNumber(1)};
        a = new CVector(aEntries);
        assertFalse(a.isOnes());

        // ------------------ Sub-case 5 ------------------
        aEntries = new CNumber[]{new CNumber(1), new CNumber(5.3, 91.3), new CNumber(1)};
        a = new CVector(aEntries);
        assertFalse(a.isOnes());

        // ------------------ Sub-case 6 ------------------
        aEntries = new CNumber[]{new CNumber(1), new CNumber(1), new CNumber(1, -1)};
        a = new CVector(aEntries);
        assertFalse(a.isOnes());
    }
}
