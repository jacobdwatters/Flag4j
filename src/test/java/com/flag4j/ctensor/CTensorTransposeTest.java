package com.flag4j.ctensor;

import com.flag4j.CTensor;
import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CTensorTransposeTest {
    static int[] aAxes;

    static CNumber[] aEntries;
    static CNumber[] expEntries;

    static Shape aShape;
    static Shape expShape;

    static CTensor A;
    static CTensor exp;

    @BeforeEach
    void setup() {
        aEntries = new CNumber[]{
                new CNumber("1.4415+9.14i"), new CNumber("235.61-9.865i"), new CNumber("-0.00024+5.15i"),
                new CNumber("1.0"), new CNumber("-0.0-85.1i"), new CNumber("1.345+3.5i"),
                new CNumber("0.014+0.01i"), new CNumber("-140.0-0.0235i"), new CNumber("1.5+9.24i"),
                new CNumber("51.0"), new CNumber("6.1-265.55i"), new CNumber("-0.00014+4.14i")};
        aShape = new Shape(3, 2, 1, 2);
        A = new CTensor(aShape, aEntries);
    }


    @Test
    void transposeTest() {
        // -------------------- Sub-case 1 --------------------
        expEntries = new CNumber[]{
                new CNumber("1.4415+9.14i"), new CNumber("-0.0-85.1i"), new CNumber("1.5+9.24i"),
                new CNumber("-0.00024+5.15i"), new CNumber("0.014+0.01i"), new CNumber("6.1-265.55i"),
                new CNumber("235.61-9.865i"), new CNumber("1.345+3.5i"), new CNumber("51.0"),
                new CNumber("1.0"), new CNumber("-140.0-0.0235i"), new CNumber("-0.00014+4.14i")};
        expShape = new Shape(2, 2, 1, 3);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.transpose());

        // -------------------- Sub-case 2 --------------------
        aAxes = new int[]{0, 2, 3, 1};
        expEntries = new CNumber[]{
                new CNumber("1.4415+9.14i"), new CNumber("-0.00024+5.15i"), new CNumber("235.61-9.865i"),
                new CNumber("1.0"), new CNumber("-0.0-85.1i"), new CNumber("0.014+0.01i"),
                new CNumber("1.345+3.5i"), new CNumber("-140.0-0.0235i"), new CNumber("1.5+9.24i"),
                new CNumber("6.1-265.55i"), new CNumber("51.0"), new CNumber("-0.00014+4.14i")};
        expShape = new Shape(3, 1, 2, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.transpose(aAxes));

        // -------------------- Sub-case 3 --------------------
        aAxes = new int[]{3, 2, 1, 0};
        expEntries = new CNumber[]{
                new CNumber("1.4415+9.14i"), new CNumber("-0.0-85.1i"), new CNumber("1.5+9.24i"),
                new CNumber("-0.00024+5.15i"), new CNumber("0.014+0.01i"), new CNumber("6.1-265.55i"),
                new CNumber("235.61-9.865i"), new CNumber("1.345+3.5i"), new CNumber("51.0"),
                new CNumber("1.0"), new CNumber("-140.0-0.0235i"), new CNumber("-0.00014+4.14i")};
        expShape = new Shape(2, 1, 2, 3);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.transpose(aAxes));

        // -------------------- Sub-case 4 --------------------
        expEntries = new CNumber[]{
                new CNumber("1.4415+9.14i"), new CNumber("-0.00024+5.15i"), new CNumber("235.61-9.865i"),
                new CNumber("1.0"), new CNumber("-0.0-85.1i"), new CNumber("0.014+0.01i"),
                new CNumber("1.345+3.5i"), new CNumber("-140.0-0.0235i"), new CNumber("1.5+9.24i"),
                new CNumber("6.1-265.55i"), new CNumber("51.0"), new CNumber("-0.00014+4.14i")};
        expShape = new Shape(3, 2, 1, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.transpose(3, 1));

        // -------------------- Sub-case 5 --------------------
        aAxes = new int[]{3, 2, 1, 2};
        assertThrows(IllegalArgumentException.class, ()->A.transpose(aAxes));

        // -------------------- Sub-case 6 --------------------
        aAxes = new int[]{3, 2, 1};
        assertThrows(IllegalArgumentException.class, ()->A.transpose(aAxes));

        // -------------------- Sub-case 7 --------------------
        aAxes = new int[]{0, 1, 3, 2, 4};
        assertThrows(IllegalArgumentException.class, ()->A.transpose(aAxes));

        // -------------------- Sub-case 8 --------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.transpose(-1, 0));

        // -------------------- Sub-case 9 --------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.transpose(1, 6));
    }
}
