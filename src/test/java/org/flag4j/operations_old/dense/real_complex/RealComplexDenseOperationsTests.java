package org.flag4j.operations_old.dense.real_complex;


import org.flag4j.complex_numbers.CNumber;
import org.flag4j.arrays.Shape;
import org.flag4j.operations_old.common.complex.ComplexOperations;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.flag4j.operations_old.dense.real_complex.RealComplexDenseOperations.add;
import static org.flag4j.operations_old.dense.real_complex.RealComplexDenseOperations.sub;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RealComplexDenseOperationsTests {
    CNumber[] src1, expResult;
    double[] src2;
    double a;
    CNumber aC;
    Shape shape1, shape2;

    @Test
    void addTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new double[]{1.1345, 2, -3, 4.13};
        expResult = new CNumber[]{new CNumber(9+1.1345, -1), new CNumber(-0.99+2, 13.445),
                new CNumber(0.9133-3), new CNumber(0+4.13, 10.3)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertArrayEquals(expResult, add(src1, shape1, src2, shape2));

        // ---------- Sub-case 2 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new double[]{1, 2, 3, 4, 5};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(LinearAlgebraException.class, () -> add(src1, shape1, src2, shape2));

        // ---------- Sub-case 3 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new double[]{1, 2};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(LinearAlgebraException.class, () -> add(src1, shape1, src2, shape2));
    }


    @Test
    void addScaleComplex() {
        // ---------- Sub-case 1 -----------------
        src2 = new double[]{1.1345, 2, -3, 4.13};
        aC = new CNumber(10.34, -9.331);
        expResult = new CNumber[]{new CNumber(10.34+1.1345, -9.331), new CNumber(10.34+2, -9.331),
                new CNumber(10.34-3, -9.331), new CNumber(10.34+4.13, -9.331)};
        Assertions.assertArrayEquals(expResult, RealComplexDenseVectorOperations.add(src2, aC));
    }


    @Test
    void subTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new double[]{1.1345, 2, -3, 4.13};
        expResult = new CNumber[]{new CNumber(9-1.1345, -1), new CNumber(-0.99-2, 13.445),
                new CNumber(0.9133+3), new CNumber(0-4.13, 10.3)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertArrayEquals(expResult, sub(src1, shape1, src2, shape2));

        // ---------- Sub-case 2 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new double[]{1, 2, 3, 4, 5};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(LinearAlgebraException.class, () -> sub(src1, shape1, src2, shape2));

        // ---------- Sub-case 3 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new double[]{1, 2};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(LinearAlgebraException.class, () -> sub(src1, shape1, src2, shape2));
    }

    @Test
    void subReverseTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new double[]{1.1345, 2, -3, 4.13};
        expResult = new CNumber[]{new CNumber(1.1345-9, 1), new CNumber(2+0.99, -13.445),
                new CNumber(-3-0.9133), new CNumber(4.13, -10.3)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertArrayEquals(expResult, sub(src2, shape2, src1, shape1));

        // ---------- Sub-case 2 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new double[]{1, 2, 3, 4, 5};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(LinearAlgebraException.class, () -> sub(src2, shape2, src1, shape1));

        // ---------- Sub-case 3 -----------------
        src1 = new CNumber[]{new CNumber(9, -1), new CNumber(-0.99, 13.445),
                new CNumber(0.9133), new CNumber(0, 10.3)};
        src2 = new double[]{1, 2};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(LinearAlgebraException.class, () -> sub(src2, shape2, src1, shape1));
    }


    @Test
    void subScaleComplex() {
        // ---------- Sub-case 1 -----------------
        src2 = new double[]{1.1345, 2, -3, 4.13};
        aC = new CNumber(10.34, -9.331);
        expResult = new CNumber[]{new CNumber(1.1345-10.34, 9.331), new CNumber(2-10.34, 9.331),
                new CNumber(-3-10.34, 9.331), new CNumber(4.13-10.34, 9.331)};
        assertArrayEquals(expResult, sub(src2, aC));
    }


    @Test
    void scaleMultComplex() {
        // ---------- Sub-case 1 -----------------
        src2 = new double[]{1.1345, 2, -3, 4.13};
        aC = new CNumber(10.34, -9.331);
        expResult = new CNumber[]{aC.mult(1.1345), aC.mult(2),
                aC.mult(-3), aC.mult(4.13)};
        assertArrayEquals(expResult, ComplexOperations.scalMult(src2, aC));
    }
}
