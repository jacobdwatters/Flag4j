package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SparseCVectorElemDivTests {

    SparseCVector a;
    int size;


    @Test
    void denseElemDivTest() {
        CNumber[] aValues = {
                new CNumber(1.3345, -9.25), new CNumber(0, -45.62),
                new CNumber(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new SparseCVector(size, aValues, aIndices);

        double[] bValues;
        int[] expIndices;
        Vector b;
        CNumber[] expValues;
        SparseCVector exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51, 0.14};
        b = new Vector(bValues);

        expValues = new CNumber[]{new CNumber(1.3345, -9.25).div(1.223), new CNumber(0, -45.62).div(3.4),
                new CNumber(25.612, 0.0245).div(-14.51)};
        expIndices = new int[]{0, 2, 5};
        exp = new SparseCVector(size, expValues, expIndices);
        assertEquals(exp, a.elemDiv(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51};
        b = new Vector(bValues);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemDiv(finalB));
    }


    @Test
    void denseComplexElemDivTest() {
        CNumber[] aValues = {
                new CNumber(1.3345, -9.25), new CNumber(0, -45.62),
                new CNumber(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new SparseCVector(size, aValues, aIndices);

        CNumber[] bValues, expValues;
        int[] expIndices;
        CVector b;
        SparseCVector exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new CNumber[]{new CNumber(24.3, -0.013), new CNumber(0, 13.6),
                new CNumber(2.4), new CNumber(-994.1 ,1.45), new CNumber(1495, 13.4),
                new CNumber(9924.515, 51.5), new CNumber(24.56, -88.351)};
        b = new CVector(bValues);

        expValues = new CNumber[]{
                new CNumber(1.3345, -9.25).div(new CNumber(24.3, -0.013)),
                new CNumber(0, -45.62).div(new CNumber(2.4)),
                new CNumber(25.612, 0.0245).div(new CNumber(9924.515, 51.5))};
        expIndices = new int[]{0, 2, 5};
        exp = new SparseCVector(size, expValues, expIndices);
        assertEquals(exp, a.elemDiv(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new CNumber[]{new CNumber(24.3, -0.013), new CNumber(0, 13.6), new CNumber(24),
                new CNumber(2.4), new CNumber(-994.1 ,1.45), new CNumber(1495, 13.4)};
        b = new CVector(bValues);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemDiv(finalB));
    }


    @Test
    void complexScalarDivTest() {
        CNumber[] aValues = {
                new CNumber(1.3345, -9.25), new CNumber(0, -45.62),
                new CNumber(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new SparseCVector(size, aValues, aIndices);

        CNumber[] expValues;
        int[] expIndices;
        CNumber b;
        SparseCVector exp;

        // -------------------- Sub-case 1 --------------------
        b = new CNumber(23.55, -984.2);

        expValues = new CNumber[]{
                new CNumber(1.3345, -9.25).div(b),
                new CNumber(0, -45.62).div(b),
                new CNumber(25.612, 0.0245).div(b)};
        expIndices = new int[]{0, 2, 5};
        exp = new SparseCVector(size, expValues, expIndices);
        assertEquals(exp, a.div(b));
    }


    @Test
    void realScalarDivTest() {
        CNumber[] aValues = {
                new CNumber(1.3345, -9.25), new CNumber(0, -45.62),
                new CNumber(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new SparseCVector(size, aValues, aIndices);

        CNumber[] expValues;
        int[] expIndices;
        double b;
        SparseCVector exp;

        // -------------------- Sub-case 1 --------------------
        b = 24.5;

        expValues = new CNumber[]{
                new CNumber(1.3345, -9.25).div(b),
                new CNumber(0, -45.62).div(b),
                new CNumber(25.612, 0.0245).div(b)};
        expIndices = new int[]{0, 2, 5};
        exp = new SparseCVector(size, expValues, expIndices);
        assertEquals(exp, a.div(b));
    }
}
