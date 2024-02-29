package org.flag4j.complex_sparse_matrix;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.dense.CVector;
import org.flag4j.sparse.CooCMatrix;
import org.flag4j.sparse.CooCVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixSetColTests {

    @Test
    void setColTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bIndices;
        CNumber[] bEntries;
        CooCVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.8969+0.65632i"), new CNumber("0.44955+0.52831i"), new CNumber("0.61332+0.4159i")};
        aRowIndices = new int[]{0, 0, 0};
        aColIndices = new int[]{0, 1, 4};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new CNumber[]{new CNumber("0.7937+0.18442i")};
        bIndices = new int[]{1};
        b = new CooCVector(bShape.get(0), bEntries, bIndices);

        expShape = new Shape(3, 5);
        expEntries = new CNumber[]{new CNumber("0.44955+0.52831i"), new CNumber("0.61332+0.4159i"), new CNumber("0.7937+0.18442i")};
        expRowIndices = new int[]{0, 0, 1};
        expColIndices = new int[]{1, 4, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new CNumber[]{new CNumber("0.14783+0.979i"), new CNumber("0.24607+0.9379i"), new CNumber("0.59171+0.21038i"), new CNumber("0.08942+0.19028i"), new CNumber("0.3154+0.99695i")};
        aRowIndices = new int[]{1, 3, 7, 9, 10};
        aColIndices = new int[]{8, 11, 9, 3, 17};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(11);
        bEntries = new CNumber[]{new CNumber("0.27382+0.15016i"), new CNumber("0.71287+0.96394i"), new CNumber("0.02+0.81414i"), new CNumber("0.34607+0.36712i")};
        bIndices = new int[]{0, 2, 7, 10};
        b = new CooCVector(bShape.get(0), bEntries, bIndices);

        expShape = new Shape(11, 23);
        expEntries = new CNumber[]{new CNumber("0.27382+0.15016i"), new CNumber("0.14783+0.979i"), new CNumber("0.71287+0.96394i"), new CNumber("0.24607+0.9379i"), new CNumber("0.02+0.81414i"), new CNumber("0.59171+0.21038i"), new CNumber("0.08942+0.19028i"), new CNumber("0.34607+0.36712i"), new CNumber("0.3154+0.99695i")};
        expRowIndices = new int[]{0, 1, 2, 3, 7, 7, 9, 10, 10};
        expColIndices = new int[]{6, 8, 6, 11, 6, 9, 3, 6, 17};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 6));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new CNumber[]{new CNumber("0.40108+0.20157i"), new CNumber("0.06329+0.35963i"), new CNumber("0.30048+0.19493i"), new CNumber("0.77913+0.49546i"), new CNumber("0.44026+0.50983i"), new CNumber("0.8898+0.04999i"), new CNumber("0.76201+0.8372i"), new CNumber("0.4491+0.55776i"), new CNumber("0.33349+0.84622i"), new CNumber("0.53084+0.75426i"), new CNumber("0.8051+0.60392i"), new CNumber("0.24584+0.2645i"), new CNumber("0.00087+0.6845i"), new CNumber("0.84881+0.82459i"), new CNumber("0.74433+0.5689i"), new CNumber("0.50837+0.7612i"), new CNumber("0.21847+0.62454i"), new CNumber("0.46747+0.08047i"), new CNumber("0.99323+0.17234i"), new CNumber("0.09636+0.99651i"), new CNumber("0.73467+0.27681i"), new CNumber("0.59151+0.41083i"), new CNumber("0.64217+0.61058i"), new CNumber("0.18733+0.53352i"), new CNumber("0.24633+0.43535i"), new CNumber("0.39783+0.69325i"), new CNumber("0.18177+0.05664i"), new CNumber("0.56967+0.42902i"), new CNumber("0.43933+0.16761i"), new CNumber("0.51236+0.47152i"), new CNumber("0.15577+0.02981i"), new CNumber("0.04133+0.00795i"), new CNumber("0.45155+0.55251i"), new CNumber("0.91551+0.41024i"), new CNumber("0.92948+0.36421i"), new CNumber("0.85228+0.0383i"), new CNumber("0.87495+0.62375i"), new CNumber("0.96061+0.89504i"), new CNumber("0.44373+0.21953i"), new CNumber("0.80214+0.71303i"), new CNumber("0.66831+0.20498i"), new CNumber("0.52174+0.18296i"), new CNumber("0.54523+0.20421i"), new CNumber("0.23072+0.94682i"), new CNumber("0.20653+0.2705i"), new CNumber("0.78021+0.97566i"), new CNumber("0.48083+0.21072i"), new CNumber("0.24483+0.23042i"), new CNumber("0.34011+0.34358i"), new CNumber("0.19875+0.75521i"), new CNumber("0.28207+0.69059i"), new CNumber("0.47211+0.12439i"), new CNumber("0.87843+0.54793i"), new CNumber("0.22343+0.3172i"), new CNumber("0.65505+0.6357i"), new CNumber("0.91364+0.00219i"), new CNumber("0.96748+0.72511i"), new CNumber("0.36055+0.92089i"), new CNumber("0.57674+0.12894i"), new CNumber("0.78033+0.60651i"), new CNumber("0.00551+0.2861i"), new CNumber("0.98859+0.32837i"), new CNumber("0.53356+0.61318i"), new CNumber("0.92697+0.67302i"), new CNumber("0.27655+0.26212i"), new CNumber("0.39756+0.94816i"), new CNumber("0.17353+0.80238i"), new CNumber("0.55777+0.74899i"), new CNumber("0.06685+0.24266i"), new CNumber("0.05017+0.32321i"), new CNumber("0.8217+0.15646i"), new CNumber("0.7867+0.71882i"), new CNumber("0.19771+0.67689i"), new CNumber("0.39419+0.54151i"), new CNumber("0.25364+0.28836i"), new CNumber("0.51256+0.28559i"), new CNumber("0.90397+0.5678i"), new CNumber("0.6858+0.47496i"), new CNumber("0.45589+0.43801i"), new CNumber("0.64739+0.01515i"), new CNumber("0.36916+0.57014i"), new CNumber("0.35399+0.27049i"), new CNumber("0.25753+0.73246i"), new CNumber("0.24983+0.05406i"), new CNumber("0.51121+0.45062i"), new CNumber("0.39576+0.64992i"), new CNumber("0.19051+0.97617i"), new CNumber("0.41709+0.90475i"), new CNumber("0.47649+0.10058i"), new CNumber("0.55394+0.99386i"), new CNumber("0.04258+0.47874i"), new CNumber("0.48511+0.53381i"), new CNumber("0.5789+0.8816i"), new CNumber("0.1921+0.32532i"), new CNumber("0.48501+0.36391i"), new CNumber("0.43698+0.97725i"), new CNumber("0.75557+0.5387i"), new CNumber("0.56365+0.7452i")};
        aRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        aColIndices = new int[]{107, 118, 156, 259, 261, 306, 447, 455, 525, 599, 623, 724, 797, 903, 929, 967, 974, 989, 43, 59, 100, 104, 126, 129, 184, 267, 290, 312, 482, 486, 487, 620, 676, 683, 895, 896, 912, 932, 941, 952, 39, 71, 103, 177, 226, 256, 367, 381, 392, 422, 569, 588, 590, 608, 619, 669, 672, 687, 707, 763, 824, 891, 921, 67, 155, 190, 250, 258, 290, 301, 343, 417, 455, 488, 504, 578, 617, 792, 818, 842, 974, 68, 87, 144, 280, 329, 392, 445, 475, 478, 479, 497, 534, 550, 628, 718, 868, 982};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new CNumber[]{new CNumber("0.73879+0.06946i"), new CNumber("0.45811+0.14543i")};
        bIndices = new int[]{0, 4};
        b = new CooCVector(bShape.get(0), bEntries, bIndices);

        expShape = new Shape(5, 1000);
        expEntries = new CNumber[]{new CNumber("0.40108+0.20157i"), new CNumber("0.06329+0.35963i"), new CNumber("0.30048+0.19493i"), new CNumber("0.77913+0.49546i"), new CNumber("0.44026+0.50983i"), new CNumber("0.8898+0.04999i"), new CNumber("0.76201+0.8372i"), new CNumber("0.4491+0.55776i"), new CNumber("0.33349+0.84622i"), new CNumber("0.53084+0.75426i"), new CNumber("0.8051+0.60392i"), new CNumber("0.24584+0.2645i"), new CNumber("0.00087+0.6845i"), new CNumber("0.84881+0.82459i"), new CNumber("0.74433+0.5689i"), new CNumber("0.50837+0.7612i"), new CNumber("0.21847+0.62454i"), new CNumber("0.46747+0.08047i"), new CNumber("0.73879+0.06946i"), new CNumber("0.99323+0.17234i"), new CNumber("0.09636+0.99651i"), new CNumber("0.73467+0.27681i"), new CNumber("0.59151+0.41083i"), new CNumber("0.64217+0.61058i"), new CNumber("0.18733+0.53352i"), new CNumber("0.24633+0.43535i"), new CNumber("0.39783+0.69325i"), new CNumber("0.18177+0.05664i"), new CNumber("0.56967+0.42902i"), new CNumber("0.43933+0.16761i"), new CNumber("0.51236+0.47152i"), new CNumber("0.15577+0.02981i"), new CNumber("0.04133+0.00795i"), new CNumber("0.45155+0.55251i"), new CNumber("0.91551+0.41024i"), new CNumber("0.92948+0.36421i"), new CNumber("0.85228+0.0383i"), new CNumber("0.87495+0.62375i"), new CNumber("0.96061+0.89504i"), new CNumber("0.44373+0.21953i"), new CNumber("0.80214+0.71303i"), new CNumber("0.66831+0.20498i"), new CNumber("0.52174+0.18296i"), new CNumber("0.54523+0.20421i"), new CNumber("0.23072+0.94682i"), new CNumber("0.20653+0.2705i"), new CNumber("0.78021+0.97566i"), new CNumber("0.48083+0.21072i"), new CNumber("0.24483+0.23042i"), new CNumber("0.34011+0.34358i"), new CNumber("0.19875+0.75521i"), new CNumber("0.28207+0.69059i"), new CNumber("0.47211+0.12439i"), new CNumber("0.87843+0.54793i"), new CNumber("0.22343+0.3172i"), new CNumber("0.65505+0.6357i"), new CNumber("0.91364+0.00219i"), new CNumber("0.96748+0.72511i"), new CNumber("0.36055+0.92089i"), new CNumber("0.57674+0.12894i"), new CNumber("0.78033+0.60651i"), new CNumber("0.00551+0.2861i"), new CNumber("0.98859+0.32837i"), new CNumber("0.53356+0.61318i"), new CNumber("0.92697+0.67302i"), new CNumber("0.27655+0.26212i"), new CNumber("0.39756+0.94816i"), new CNumber("0.17353+0.80238i"), new CNumber("0.55777+0.74899i"), new CNumber("0.06685+0.24266i"), new CNumber("0.05017+0.32321i"), new CNumber("0.8217+0.15646i"), new CNumber("0.7867+0.71882i"), new CNumber("0.19771+0.67689i"), new CNumber("0.39419+0.54151i"), new CNumber("0.25364+0.28836i"), new CNumber("0.51256+0.28559i"), new CNumber("0.90397+0.5678i"), new CNumber("0.6858+0.47496i"), new CNumber("0.45589+0.43801i"), new CNumber("0.64739+0.01515i"), new CNumber("0.36916+0.57014i"), new CNumber("0.35399+0.27049i"), new CNumber("0.25753+0.73246i"), new CNumber("0.24983+0.05406i"), new CNumber("0.51121+0.45062i"), new CNumber("0.39576+0.64992i"), new CNumber("0.19051+0.97617i"), new CNumber("0.41709+0.90475i"), new CNumber("0.47649+0.10058i"), new CNumber("0.55394+0.99386i"), new CNumber("0.04258+0.47874i"), new CNumber("0.48511+0.53381i"), new CNumber("0.5789+0.8816i"), new CNumber("0.1921+0.32532i"), new CNumber("0.48501+0.36391i"), new CNumber("0.43698+0.97725i"), new CNumber("0.75557+0.5387i"), new CNumber("0.56365+0.7452i"), new CNumber("0.45811+0.14543i")};
        expRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        expColIndices = new int[]{107, 118, 156, 259, 261, 306, 447, 455, 525, 599, 623, 724, 797, 903, 929, 967, 974, 989, 999, 43, 59, 100, 104, 126, 129, 184, 267, 290, 312, 482, 486, 487, 620, 676, 683, 895, 896, 912, 932, 941, 952, 39, 71, 103, 177, 226, 256, 367, 381, 392, 422, 569, 588, 590, 608, 619, 669, 672, 687, 707, 763, 824, 891, 921, 67, 155, 190, 250, 258, 290, 301, 343, 417, 455, 488, 504, 578, 617, 792, 818, 842, 974, 68, 87, 144, 280, 329, 392, 445, 475, 478, 479, 497, 534, 550, 628, 718, 868, 982, 999};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 999));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.70609+0.77623i"), new CNumber("0.61771+0.37191i"), new CNumber("0.93683+0.4622i"), new CNumber("0.22006+0.06678i")};
        aRowIndices = new int[]{1, 1, 2, 4};
        aColIndices = new int[]{0, 1, 1, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new CNumber[]{new CNumber("0.26721+0.56832i"), new CNumber("0.09961+0.57491i")};
        bIndices = new int[]{2, 3};
        b = new CooCVector(bShape.get(0), bEntries, bIndices);

        CooCMatrix final0a = a;
        CooCVector final0b = b;
        assertThrows(Exception.class, ()->final0a.setCol(final0b, -1));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.3543+0.92325i"), new CNumber("0.01246+0.59718i"), new CNumber("0.65951+0.75695i"), new CNumber("0.5218+0.61097i")};
        aRowIndices = new int[]{1, 2, 2, 4};
        aColIndices = new int[]{1, 0, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new CNumber[]{new CNumber("0.26676+0.40491i"), new CNumber("0.41776+0.60048i")};
        bIndices = new int[]{1, 2};
        b = new CooCVector(bShape.get(0), bEntries, bIndices);

        CooCMatrix final1a = a;
        CooCVector final1b = b;
        assertThrows(Exception.class, ()->final1a.setCol(final1b, 4));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.44339+0.35944i"), new CNumber("0.9204+0.18754i"), new CNumber("0.17881+0.76557i"), new CNumber("0.60444+0.86329i")};
        aRowIndices = new int[]{0, 3, 3, 4};
        aColIndices = new int[]{1, 0, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(7);
        bEntries = new CNumber[]{new CNumber("0.29086+0.80883i"), new CNumber("0.05468+0.08505i")};
        bIndices = new int[]{0, 2};
        b = new CooCVector(bShape.get(0), bEntries, bIndices);

        CooCMatrix final2a = a;
        CooCVector final2b = b;
        assertThrows(Exception.class, ()->final2a.setCol(final2b, 0));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.5686+0.2999i"), new CNumber("0.86547+0.85179i"), new CNumber("0.66343+0.4857i"), new CNumber("0.84483+0.58998i")};
        aRowIndices = new int[]{0, 1, 3, 3};
        aColIndices = new int[]{2, 2, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new CNumber[]{new CNumber("0.56015+0.91847i")};
        bIndices = new int[]{1};
        b = new CooCVector(bShape.get(0), bEntries, bIndices);

        CooCMatrix final3a = a;
        CooCVector final3b = b;
        assertThrows(Exception.class, ()->final3a.setCol(final3b, 0));
    }


    @Test
    void setColDenseTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        CNumber[] bEntries;
        CVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.78423+0.54028i"), new CNumber("0.76636+0.06643i"), new CNumber("0.36247+0.98024i")};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{0, 2, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.97768+0.537i"), new CNumber("0.49928+0.5531i"), new CNumber("0.78974+0.27284i")};
        b = new CVector(bEntries);

        expShape = new Shape(3, 5);
        expEntries = new CNumber[]{new CNumber("0.97768+0.537i"), new CNumber("0.76636+0.06643i"), new CNumber("0.49928+0.5531i"), new CNumber("0.36247+0.98024i"), new CNumber("0.78974+0.27284i")};
        expRowIndices = new int[]{0, 0, 1, 1, 2};
        expColIndices = new int[]{0, 2, 0, 1, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new CNumber[]{new CNumber("0.07992+0.89737i"), new CNumber("0.13427+0.62342i"), new CNumber("0.28456+0.46146i"), new CNumber("0.60407+0.27963i"), new CNumber("0.04068+0.95748i")};
        aRowIndices = new int[]{2, 5, 5, 6, 9};
        aColIndices = new int[]{17, 1, 12, 20, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.77962+0.32757i"), new CNumber("0.43456+0.2328i"), new CNumber("0.81963+0.31615i"), new CNumber("0.07205+0.88257i"), new CNumber("0.95017+0.25251i"), new CNumber("0.31793+0.78868i"), new CNumber("0.61734+0.80147i"), new CNumber("0.37761+0.36171i"), new CNumber("0.25995+0.85692i"), new CNumber("0.72975+0.29766i"), new CNumber("0.70145+0.52637i")};
        b = new CVector(bEntries);

        expShape = new Shape(11, 23);
        expEntries = new CNumber[]{new CNumber("0.77962+0.32757i"), new CNumber("0.43456+0.2328i"), new CNumber("0.81963+0.31615i"), new CNumber("0.07992+0.89737i"), new CNumber("0.07205+0.88257i"), new CNumber("0.95017+0.25251i"), new CNumber("0.13427+0.62342i"), new CNumber("0.31793+0.78868i"), new CNumber("0.28456+0.46146i"), new CNumber("0.61734+0.80147i"), new CNumber("0.60407+0.27963i"), new CNumber("0.37761+0.36171i"), new CNumber("0.25995+0.85692i"), new CNumber("0.04068+0.95748i"), new CNumber("0.72975+0.29766i"), new CNumber("0.70145+0.52637i")};
        expRowIndices = new int[]{0, 1, 2, 2, 3, 4, 5, 5, 5, 6, 6, 7, 8, 9, 9, 10};
        expColIndices = new int[]{6, 6, 6, 17, 6, 6, 1, 6, 12, 6, 20, 6, 6, 1, 6, 6};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 6));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new CNumber[]{new CNumber("0.5467+0.49999i"), new CNumber("0.37116+0.44247i"), new CNumber("0.81561+0.9467i"), new CNumber("0.34933+0.69447i"), new CNumber("0.39814+0.90027i"), new CNumber("0.47298+0.73093i"), new CNumber("0.61647+0.03775i"), new CNumber("0.05963+0.4484i"), new CNumber("0.24081+0.97427i"), new CNumber("0.67862+0.36167i"), new CNumber("0.02861+0.69044i"), new CNumber("0.00907+0.99155i"), new CNumber("0.65753+0.3947i"), new CNumber("0.59853+0.3522i"), new CNumber("0.35532+0.51502i"), new CNumber("0.39197+0.29002i"), new CNumber("0.52439+0.44036i"), new CNumber("0.05276+0.45663i"), new CNumber("0.3425+0.80646i"), new CNumber("0.14171+0.41399i"), new CNumber("0.6852+0.76258i"), new CNumber("0.51293+0.51815i"), new CNumber("0.07046+0.7277i"), new CNumber("0.9778+0.92829i"), new CNumber("0.18135+0.61364i"), new CNumber("0.0224+0.6758i"), new CNumber("0.52499+0.03964i"), new CNumber("0.66218+0.9008i"), new CNumber("0.10323+0.78195i"), new CNumber("0.21798+0.35071i"), new CNumber("0.40509+0.81845i"), new CNumber("0.14194+0.33584i"), new CNumber("0.11069+0.41262i"), new CNumber("0.04064+0.87248i"), new CNumber("0.42594+0.99876i"), new CNumber("0.03525+0.73718i"), new CNumber("0.48781+0.54168i"), new CNumber("0.17025+0.03177i"), new CNumber("0.48174+0.24292i"), new CNumber("0.04537+0.74363i"), new CNumber("0.66061+0.39804i"), new CNumber("0.67876+0.45574i"), new CNumber("0.69043+0.81036i"), new CNumber("0.11492+0.87021i"), new CNumber("0.01063+0.07606i"), new CNumber("0.59802+0.92673i"), new CNumber("0.36591+0.97066i"), new CNumber("0.62076+0.89358i"), new CNumber("0.55133+0.34117i"), new CNumber("0.5385+0.07551i"), new CNumber("0.20212+0.92532i"), new CNumber("0.7194+0.17267i"), new CNumber("0.17862+0.66152i"), new CNumber("0.49722+0.98283i"), new CNumber("0.26972+0.01576i"), new CNumber("0.29695+0.13388i"), new CNumber("0.05378+0.96254i"), new CNumber("0.57821+0.71101i"), new CNumber("0.72685+0.52017i"), new CNumber("0.38257+0.81359i"), new CNumber("0.76604+0.77728i"), new CNumber("0.89312+0.64907i"), new CNumber("0.49444+0.81506i"), new CNumber("0.65258+0.38833i"), new CNumber("0.16306+0.83022i"), new CNumber("0.50171+0.38688i"), new CNumber("0.92758+0.59179i"), new CNumber("0.2291+0.57235i"), new CNumber("0.63209+0.29828i"), new CNumber("0.14085+0.73637i"), new CNumber("0.06266+0.74583i"), new CNumber("0.35524+0.51969i"), new CNumber("0.0463+0.52071i"), new CNumber("0.44881+0.29868i"), new CNumber("0.62168+0.37864i"), new CNumber("0.53248+0.91364i"), new CNumber("0.98381+0.68962i"), new CNumber("0.50729+0.37385i"), new CNumber("0.80509+0.73895i"), new CNumber("0.07097+0.58097i"), new CNumber("0.93424+0.19568i"), new CNumber("0.23747+0.23004i"), new CNumber("0.15703+0.5813i"), new CNumber("0.1662+0.69171i"), new CNumber("0.37488+0.23289i"), new CNumber("0.67413+0.2455i"), new CNumber("0.27175+0.0791i"), new CNumber("0.95064+0.27993i"), new CNumber("0.38653+0.17386i"), new CNumber("0.68125+0.85424i"), new CNumber("0.17445+0.07739i"), new CNumber("0.05146+0.778i"), new CNumber("0.99115+0.84599i"), new CNumber("0.04699+0.79527i"), new CNumber("0.3485+0.04623i"), new CNumber("0.68152+0.64108i"), new CNumber("0.38547+0.44214i"), new CNumber("0.0787+0.33698i")};
        aRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        aColIndices = new int[]{12, 30, 41, 66, 87, 107, 118, 127, 212, 306, 358, 425, 451, 484, 640, 694, 712, 787, 38, 90, 216, 293, 388, 389, 437, 453, 463, 474, 511, 564, 797, 804, 811, 896, 965, 95, 125, 291, 345, 389, 422, 482, 507, 646, 703, 749, 789, 800, 818, 963, 977, 266, 344, 491, 594, 689, 696, 705, 719, 723, 725, 768, 792, 800, 824, 827, 832, 839, 939, 962, 977, 986, 24, 47, 99, 165, 261, 273, 311, 327, 357, 361, 391, 437, 443, 528, 551, 583, 586, 766, 788, 879, 887, 904, 959, 963, 967, 982};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.40212+0.71141i"), new CNumber("0.53039+0.68416i"), new CNumber("0.72961+0.75094i"), new CNumber("0.40268+0.84779i"), new CNumber("0.44872+0.82616i")};
        b = new CVector(bEntries);

        expShape = new Shape(5, 1000);
        expEntries = new CNumber[]{new CNumber("0.5467+0.49999i"), new CNumber("0.37116+0.44247i"), new CNumber("0.81561+0.9467i"), new CNumber("0.34933+0.69447i"), new CNumber("0.39814+0.90027i"), new CNumber("0.47298+0.73093i"), new CNumber("0.61647+0.03775i"), new CNumber("0.05963+0.4484i"), new CNumber("0.24081+0.97427i"), new CNumber("0.67862+0.36167i"), new CNumber("0.02861+0.69044i"), new CNumber("0.00907+0.99155i"), new CNumber("0.65753+0.3947i"), new CNumber("0.59853+0.3522i"), new CNumber("0.35532+0.51502i"), new CNumber("0.39197+0.29002i"), new CNumber("0.52439+0.44036i"), new CNumber("0.05276+0.45663i"), new CNumber("0.40212+0.71141i"), new CNumber("0.3425+0.80646i"), new CNumber("0.14171+0.41399i"), new CNumber("0.6852+0.76258i"), new CNumber("0.51293+0.51815i"), new CNumber("0.07046+0.7277i"), new CNumber("0.9778+0.92829i"), new CNumber("0.18135+0.61364i"), new CNumber("0.0224+0.6758i"), new CNumber("0.52499+0.03964i"), new CNumber("0.66218+0.9008i"), new CNumber("0.10323+0.78195i"), new CNumber("0.21798+0.35071i"), new CNumber("0.40509+0.81845i"), new CNumber("0.14194+0.33584i"), new CNumber("0.11069+0.41262i"), new CNumber("0.04064+0.87248i"), new CNumber("0.42594+0.99876i"), new CNumber("0.53039+0.68416i"), new CNumber("0.03525+0.73718i"), new CNumber("0.48781+0.54168i"), new CNumber("0.17025+0.03177i"), new CNumber("0.48174+0.24292i"), new CNumber("0.04537+0.74363i"), new CNumber("0.66061+0.39804i"), new CNumber("0.67876+0.45574i"), new CNumber("0.69043+0.81036i"), new CNumber("0.11492+0.87021i"), new CNumber("0.01063+0.07606i"), new CNumber("0.59802+0.92673i"), new CNumber("0.36591+0.97066i"), new CNumber("0.62076+0.89358i"), new CNumber("0.55133+0.34117i"), new CNumber("0.5385+0.07551i"), new CNumber("0.20212+0.92532i"), new CNumber("0.72961+0.75094i"), new CNumber("0.7194+0.17267i"), new CNumber("0.17862+0.66152i"), new CNumber("0.49722+0.98283i"), new CNumber("0.26972+0.01576i"), new CNumber("0.29695+0.13388i"), new CNumber("0.05378+0.96254i"), new CNumber("0.57821+0.71101i"), new CNumber("0.72685+0.52017i"), new CNumber("0.38257+0.81359i"), new CNumber("0.76604+0.77728i"), new CNumber("0.89312+0.64907i"), new CNumber("0.49444+0.81506i"), new CNumber("0.65258+0.38833i"), new CNumber("0.16306+0.83022i"), new CNumber("0.50171+0.38688i"), new CNumber("0.92758+0.59179i"), new CNumber("0.2291+0.57235i"), new CNumber("0.63209+0.29828i"), new CNumber("0.14085+0.73637i"), new CNumber("0.06266+0.74583i"), new CNumber("0.35524+0.51969i"), new CNumber("0.40268+0.84779i"), new CNumber("0.0463+0.52071i"), new CNumber("0.44881+0.29868i"), new CNumber("0.62168+0.37864i"), new CNumber("0.53248+0.91364i"), new CNumber("0.98381+0.68962i"), new CNumber("0.50729+0.37385i"), new CNumber("0.80509+0.73895i"), new CNumber("0.07097+0.58097i"), new CNumber("0.93424+0.19568i"), new CNumber("0.23747+0.23004i"), new CNumber("0.15703+0.5813i"), new CNumber("0.1662+0.69171i"), new CNumber("0.37488+0.23289i"), new CNumber("0.67413+0.2455i"), new CNumber("0.27175+0.0791i"), new CNumber("0.95064+0.27993i"), new CNumber("0.38653+0.17386i"), new CNumber("0.68125+0.85424i"), new CNumber("0.17445+0.07739i"), new CNumber("0.05146+0.778i"), new CNumber("0.99115+0.84599i"), new CNumber("0.04699+0.79527i"), new CNumber("0.3485+0.04623i"), new CNumber("0.68152+0.64108i"), new CNumber("0.38547+0.44214i"), new CNumber("0.0787+0.33698i"), new CNumber("0.44872+0.82616i")};
        expRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        expColIndices = new int[]{12, 30, 41, 66, 87, 107, 118, 127, 212, 306, 358, 425, 451, 484, 640, 694, 712, 787, 999, 38, 90, 216, 293, 388, 389, 437, 453, 463, 474, 511, 564, 797, 804, 811, 896, 965, 999, 95, 125, 291, 345, 389, 422, 482, 507, 646, 703, 749, 789, 800, 818, 963, 977, 999, 266, 344, 491, 594, 689, 696, 705, 719, 723, 725, 768, 792, 800, 824, 827, 832, 839, 939, 962, 977, 986, 999, 24, 47, 99, 165, 261, 273, 311, 327, 357, 361, 391, 437, 443, 528, 551, 583, 586, 766, 788, 879, 887, 904, 959, 963, 967, 982, 999};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 999));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.58671+0.28501i"), new CNumber("0.93231+0.49035i"), new CNumber("0.81853+0.8372i"), new CNumber("0.72574+0.2339i")};
        aRowIndices = new int[]{0, 3, 3, 4};
        aColIndices = new int[]{1, 0, 1, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.34529+0.28757i"), new CNumber("0.91101+0.29749i"), new CNumber("0.8687+0.91169i"), new CNumber("0.07982+0.35095i"), new CNumber("0.87509+0.86136i")};
        b = new CVector(bEntries);

        CooCMatrix final0a = a;
        CVector final0b = b;
        assertThrows(Exception.class, ()->final0a.setCol(final0b, -1));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.68826+0.00131i"), new CNumber("0.91143+0.5812i"), new CNumber("0.44094+0.7873i"), new CNumber("0.59887+0.59552i")};
        aRowIndices = new int[]{0, 2, 2, 4};
        aColIndices = new int[]{0, 0, 2, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.71469+0.9564i"), new CNumber("0.58723+0.41359i"), new CNumber("0.34398+0.07766i"), new CNumber("0.48027+0.18874i"), new CNumber("0.29563+0.51053i")};
        b = new CVector(bEntries);

        CooCMatrix final1a = a;
        CVector final1b = b;
        assertThrows(Exception.class, ()->final1a.setCol(final1b, 4));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.06746+0.93946i"), new CNumber("0.6437+0.31881i"), new CNumber("0.83815+0.14616i"), new CNumber("0.70053+0.01414i")};
        aRowIndices = new int[]{0, 3, 4, 4};
        aColIndices = new int[]{0, 0, 0, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.11965+0.92396i"), new CNumber("0.33666+0.0976i"), new CNumber("0.94848+0.55422i"), new CNumber("0.75313+0.96464i"), new CNumber("0.98065+0.95475i"), new CNumber("0.18058+0.09665i"), new CNumber("0.74986+0.89085i")};
        b = new CVector(bEntries);

        CooCMatrix final2a = a;
        CVector final2b = b;
        assertThrows(Exception.class, ()->final2a.setCol(final2b, 0));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.71255+0.72902i"), new CNumber("0.84164+0.18667i"), new CNumber("0.80426+0.1968i"), new CNumber("0.842+0.72446i")};
        aRowIndices = new int[]{0, 0, 3, 4};
        aColIndices = new int[]{0, 1, 2, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.95894+0.49391i"), new CNumber("0.43091+0.50491i"), new CNumber("0.88756+0.28381i")};
        b = new CVector(bEntries);

        CooCMatrix final3a = a;
        CVector final3b = b;
        assertThrows(Exception.class, ()->final3a.setCol(final3b, 0));
    }


    @Test
    void setColDenseArrayTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        CNumber[] b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.78423+0.54028i"), new CNumber("0.76636+0.06643i"), new CNumber("0.36247+0.98024i")};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{0, 2, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[]{new CNumber("0.97768+0.537i"), new CNumber("0.49928+0.5531i"), new CNumber("0.78974+0.27284i")};

        expShape = new Shape(3, 5);
        expEntries = new CNumber[]{new CNumber("0.97768+0.537i"), new CNumber("0.76636+0.06643i"), new CNumber("0.49928+0.5531i"), new CNumber("0.36247+0.98024i"), new CNumber("0.78974+0.27284i")};
        expRowIndices = new int[]{0, 0, 1, 1, 2};
        expColIndices = new int[]{0, 2, 0, 1, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new CNumber[]{new CNumber("0.07992+0.89737i"), new CNumber("0.13427+0.62342i"), new CNumber("0.28456+0.46146i"), new CNumber("0.60407+0.27963i"), new CNumber("0.04068+0.95748i")};
        aRowIndices = new int[]{2, 5, 5, 6, 9};
        aColIndices = new int[]{17, 1, 12, 20, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[]{new CNumber("0.77962+0.32757i"), new CNumber("0.43456+0.2328i"), new CNumber("0.81963+0.31615i"), new CNumber("0.07205+0.88257i"), new CNumber("0.95017+0.25251i"), new CNumber("0.31793+0.78868i"), new CNumber("0.61734+0.80147i"), new CNumber("0.37761+0.36171i"), new CNumber("0.25995+0.85692i"), new CNumber("0.72975+0.29766i"), new CNumber("0.70145+0.52637i")};

        expShape = new Shape(11, 23);
        expEntries = new CNumber[]{new CNumber("0.77962+0.32757i"), new CNumber("0.43456+0.2328i"), new CNumber("0.81963+0.31615i"), new CNumber("0.07992+0.89737i"), new CNumber("0.07205+0.88257i"), new CNumber("0.95017+0.25251i"), new CNumber("0.13427+0.62342i"), new CNumber("0.31793+0.78868i"), new CNumber("0.28456+0.46146i"), new CNumber("0.61734+0.80147i"), new CNumber("0.60407+0.27963i"), new CNumber("0.37761+0.36171i"), new CNumber("0.25995+0.85692i"), new CNumber("0.04068+0.95748i"), new CNumber("0.72975+0.29766i"), new CNumber("0.70145+0.52637i")};
        expRowIndices = new int[]{0, 1, 2, 2, 3, 4, 5, 5, 5, 6, 6, 7, 8, 9, 9, 10};
        expColIndices = new int[]{6, 6, 6, 17, 6, 6, 1, 6, 12, 6, 20, 6, 6, 1, 6, 6};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 6));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new CNumber[]{new CNumber("0.5467+0.49999i"), new CNumber("0.37116+0.44247i"), new CNumber("0.81561+0.9467i"), new CNumber("0.34933+0.69447i"), new CNumber("0.39814+0.90027i"), new CNumber("0.47298+0.73093i"), new CNumber("0.61647+0.03775i"), new CNumber("0.05963+0.4484i"), new CNumber("0.24081+0.97427i"), new CNumber("0.67862+0.36167i"), new CNumber("0.02861+0.69044i"), new CNumber("0.00907+0.99155i"), new CNumber("0.65753+0.3947i"), new CNumber("0.59853+0.3522i"), new CNumber("0.35532+0.51502i"), new CNumber("0.39197+0.29002i"), new CNumber("0.52439+0.44036i"), new CNumber("0.05276+0.45663i"), new CNumber("0.3425+0.80646i"), new CNumber("0.14171+0.41399i"), new CNumber("0.6852+0.76258i"), new CNumber("0.51293+0.51815i"), new CNumber("0.07046+0.7277i"), new CNumber("0.9778+0.92829i"), new CNumber("0.18135+0.61364i"), new CNumber("0.0224+0.6758i"), new CNumber("0.52499+0.03964i"), new CNumber("0.66218+0.9008i"), new CNumber("0.10323+0.78195i"), new CNumber("0.21798+0.35071i"), new CNumber("0.40509+0.81845i"), new CNumber("0.14194+0.33584i"), new CNumber("0.11069+0.41262i"), new CNumber("0.04064+0.87248i"), new CNumber("0.42594+0.99876i"), new CNumber("0.03525+0.73718i"), new CNumber("0.48781+0.54168i"), new CNumber("0.17025+0.03177i"), new CNumber("0.48174+0.24292i"), new CNumber("0.04537+0.74363i"), new CNumber("0.66061+0.39804i"), new CNumber("0.67876+0.45574i"), new CNumber("0.69043+0.81036i"), new CNumber("0.11492+0.87021i"), new CNumber("0.01063+0.07606i"), new CNumber("0.59802+0.92673i"), new CNumber("0.36591+0.97066i"), new CNumber("0.62076+0.89358i"), new CNumber("0.55133+0.34117i"), new CNumber("0.5385+0.07551i"), new CNumber("0.20212+0.92532i"), new CNumber("0.7194+0.17267i"), new CNumber("0.17862+0.66152i"), new CNumber("0.49722+0.98283i"), new CNumber("0.26972+0.01576i"), new CNumber("0.29695+0.13388i"), new CNumber("0.05378+0.96254i"), new CNumber("0.57821+0.71101i"), new CNumber("0.72685+0.52017i"), new CNumber("0.38257+0.81359i"), new CNumber("0.76604+0.77728i"), new CNumber("0.89312+0.64907i"), new CNumber("0.49444+0.81506i"), new CNumber("0.65258+0.38833i"), new CNumber("0.16306+0.83022i"), new CNumber("0.50171+0.38688i"), new CNumber("0.92758+0.59179i"), new CNumber("0.2291+0.57235i"), new CNumber("0.63209+0.29828i"), new CNumber("0.14085+0.73637i"), new CNumber("0.06266+0.74583i"), new CNumber("0.35524+0.51969i"), new CNumber("0.0463+0.52071i"), new CNumber("0.44881+0.29868i"), new CNumber("0.62168+0.37864i"), new CNumber("0.53248+0.91364i"), new CNumber("0.98381+0.68962i"), new CNumber("0.50729+0.37385i"), new CNumber("0.80509+0.73895i"), new CNumber("0.07097+0.58097i"), new CNumber("0.93424+0.19568i"), new CNumber("0.23747+0.23004i"), new CNumber("0.15703+0.5813i"), new CNumber("0.1662+0.69171i"), new CNumber("0.37488+0.23289i"), new CNumber("0.67413+0.2455i"), new CNumber("0.27175+0.0791i"), new CNumber("0.95064+0.27993i"), new CNumber("0.38653+0.17386i"), new CNumber("0.68125+0.85424i"), new CNumber("0.17445+0.07739i"), new CNumber("0.05146+0.778i"), new CNumber("0.99115+0.84599i"), new CNumber("0.04699+0.79527i"), new CNumber("0.3485+0.04623i"), new CNumber("0.68152+0.64108i"), new CNumber("0.38547+0.44214i"), new CNumber("0.0787+0.33698i")};
        aRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        aColIndices = new int[]{12, 30, 41, 66, 87, 107, 118, 127, 212, 306, 358, 425, 451, 484, 640, 694, 712, 787, 38, 90, 216, 293, 388, 389, 437, 453, 463, 474, 511, 564, 797, 804, 811, 896, 965, 95, 125, 291, 345, 389, 422, 482, 507, 646, 703, 749, 789, 800, 818, 963, 977, 266, 344, 491, 594, 689, 696, 705, 719, 723, 725, 768, 792, 800, 824, 827, 832, 839, 939, 962, 977, 986, 24, 47, 99, 165, 261, 273, 311, 327, 357, 361, 391, 437, 443, 528, 551, 583, 586, 766, 788, 879, 887, 904, 959, 963, 967, 982};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[]{new CNumber("0.40212+0.71141i"), new CNumber("0.53039+0.68416i"), new CNumber("0.72961+0.75094i"), new CNumber("0.40268+0.84779i"), new CNumber("0.44872+0.82616i")};

        expShape = new Shape(5, 1000);
        expEntries = new CNumber[]{new CNumber("0.5467+0.49999i"), new CNumber("0.37116+0.44247i"), new CNumber("0.81561+0.9467i"), new CNumber("0.34933+0.69447i"), new CNumber("0.39814+0.90027i"), new CNumber("0.47298+0.73093i"), new CNumber("0.61647+0.03775i"), new CNumber("0.05963+0.4484i"), new CNumber("0.24081+0.97427i"), new CNumber("0.67862+0.36167i"), new CNumber("0.02861+0.69044i"), new CNumber("0.00907+0.99155i"), new CNumber("0.65753+0.3947i"), new CNumber("0.59853+0.3522i"), new CNumber("0.35532+0.51502i"), new CNumber("0.39197+0.29002i"), new CNumber("0.52439+0.44036i"), new CNumber("0.05276+0.45663i"), new CNumber("0.40212+0.71141i"), new CNumber("0.3425+0.80646i"), new CNumber("0.14171+0.41399i"), new CNumber("0.6852+0.76258i"), new CNumber("0.51293+0.51815i"), new CNumber("0.07046+0.7277i"), new CNumber("0.9778+0.92829i"), new CNumber("0.18135+0.61364i"), new CNumber("0.0224+0.6758i"), new CNumber("0.52499+0.03964i"), new CNumber("0.66218+0.9008i"), new CNumber("0.10323+0.78195i"), new CNumber("0.21798+0.35071i"), new CNumber("0.40509+0.81845i"), new CNumber("0.14194+0.33584i"), new CNumber("0.11069+0.41262i"), new CNumber("0.04064+0.87248i"), new CNumber("0.42594+0.99876i"), new CNumber("0.53039+0.68416i"), new CNumber("0.03525+0.73718i"), new CNumber("0.48781+0.54168i"), new CNumber("0.17025+0.03177i"), new CNumber("0.48174+0.24292i"), new CNumber("0.04537+0.74363i"), new CNumber("0.66061+0.39804i"), new CNumber("0.67876+0.45574i"), new CNumber("0.69043+0.81036i"), new CNumber("0.11492+0.87021i"), new CNumber("0.01063+0.07606i"), new CNumber("0.59802+0.92673i"), new CNumber("0.36591+0.97066i"), new CNumber("0.62076+0.89358i"), new CNumber("0.55133+0.34117i"), new CNumber("0.5385+0.07551i"), new CNumber("0.20212+0.92532i"), new CNumber("0.72961+0.75094i"), new CNumber("0.7194+0.17267i"), new CNumber("0.17862+0.66152i"), new CNumber("0.49722+0.98283i"), new CNumber("0.26972+0.01576i"), new CNumber("0.29695+0.13388i"), new CNumber("0.05378+0.96254i"), new CNumber("0.57821+0.71101i"), new CNumber("0.72685+0.52017i"), new CNumber("0.38257+0.81359i"), new CNumber("0.76604+0.77728i"), new CNumber("0.89312+0.64907i"), new CNumber("0.49444+0.81506i"), new CNumber("0.65258+0.38833i"), new CNumber("0.16306+0.83022i"), new CNumber("0.50171+0.38688i"), new CNumber("0.92758+0.59179i"), new CNumber("0.2291+0.57235i"), new CNumber("0.63209+0.29828i"), new CNumber("0.14085+0.73637i"), new CNumber("0.06266+0.74583i"), new CNumber("0.35524+0.51969i"), new CNumber("0.40268+0.84779i"), new CNumber("0.0463+0.52071i"), new CNumber("0.44881+0.29868i"), new CNumber("0.62168+0.37864i"), new CNumber("0.53248+0.91364i"), new CNumber("0.98381+0.68962i"), new CNumber("0.50729+0.37385i"), new CNumber("0.80509+0.73895i"), new CNumber("0.07097+0.58097i"), new CNumber("0.93424+0.19568i"), new CNumber("0.23747+0.23004i"), new CNumber("0.15703+0.5813i"), new CNumber("0.1662+0.69171i"), new CNumber("0.37488+0.23289i"), new CNumber("0.67413+0.2455i"), new CNumber("0.27175+0.0791i"), new CNumber("0.95064+0.27993i"), new CNumber("0.38653+0.17386i"), new CNumber("0.68125+0.85424i"), new CNumber("0.17445+0.07739i"), new CNumber("0.05146+0.778i"), new CNumber("0.99115+0.84599i"), new CNumber("0.04699+0.79527i"), new CNumber("0.3485+0.04623i"), new CNumber("0.68152+0.64108i"), new CNumber("0.38547+0.44214i"), new CNumber("0.0787+0.33698i"), new CNumber("0.44872+0.82616i")};
        expRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        expColIndices = new int[]{12, 30, 41, 66, 87, 107, 118, 127, 212, 306, 358, 425, 451, 484, 640, 694, 712, 787, 999, 38, 90, 216, 293, 388, 389, 437, 453, 463, 474, 511, 564, 797, 804, 811, 896, 965, 999, 95, 125, 291, 345, 389, 422, 482, 507, 646, 703, 749, 789, 800, 818, 963, 977, 999, 266, 344, 491, 594, 689, 696, 705, 719, 723, 725, 768, 792, 800, 824, 827, 832, 839, 939, 962, 977, 986, 999, 24, 47, 99, 165, 261, 273, 311, 327, 357, 361, 391, 437, 443, 528, 551, 583, 586, 766, 788, 879, 887, 904, 959, 963, 967, 982, 999};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 999));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.58671+0.28501i"), new CNumber("0.93231+0.49035i"), new CNumber("0.81853+0.8372i"), new CNumber("0.72574+0.2339i")};
        aRowIndices = new int[]{0, 3, 3, 4};
        aColIndices = new int[]{1, 0, 1, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[]{new CNumber("0.34529+0.28757i"), new CNumber("0.91101+0.29749i"), new CNumber("0.8687+0.91169i"), new CNumber("0.07982+0.35095i"), new CNumber("0.87509+0.86136i")};

        CooCMatrix final0a = a;
        CNumber[] final0b = b;
        assertThrows(Exception.class, ()->final0a.setCol(final0b, -1));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.68826+0.00131i"), new CNumber("0.91143+0.5812i"), new CNumber("0.44094+0.7873i"), new CNumber("0.59887+0.59552i")};
        aRowIndices = new int[]{0, 2, 2, 4};
        aColIndices = new int[]{0, 0, 2, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[]{new CNumber("0.71469+0.9564i"), new CNumber("0.58723+0.41359i"), new CNumber("0.34398+0.07766i"), new CNumber("0.48027+0.18874i"), new CNumber("0.29563+0.51053i")};

        CooCMatrix final1a = a;
        CNumber[] final1b = b;
        assertThrows(Exception.class, ()->final1a.setCol(final1b, 4));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.06746+0.93946i"), new CNumber("0.6437+0.31881i"), new CNumber("0.83815+0.14616i"), new CNumber("0.70053+0.01414i")};
        aRowIndices = new int[]{0, 3, 4, 4};
        aColIndices = new int[]{0, 0, 0, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[]{new CNumber("0.11965+0.92396i"), new CNumber("0.33666+0.0976i"), new CNumber("0.94848+0.55422i"), new CNumber("0.75313+0.96464i"), new CNumber("0.98065+0.95475i"), new CNumber("0.18058+0.09665i"), new CNumber("0.74986+0.89085i")};

        CooCMatrix final2a = a;
        CNumber[] final2b = b;
        assertThrows(Exception.class, ()->final2a.setCol(final2b, 0));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.71255+0.72902i"), new CNumber("0.84164+0.18667i"), new CNumber("0.80426+0.1968i"), new CNumber("0.842+0.72446i")};
        aRowIndices = new int[]{0, 0, 3, 4};
        aColIndices = new int[]{0, 1, 2, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[]{new CNumber("0.95894+0.49391i"), new CNumber("0.43091+0.50491i"), new CNumber("0.88756+0.28381i")};

        CooCMatrix final3a = a;
        CNumber[] final3b = b;
        assertThrows(Exception.class, ()->final3a.setCol(final3b, 0));
    }


    @Test
    void setColRealDenseArrayTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        double[] b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.53552+0.98065i"), new CNumber("0.34883+0.37796i"), new CNumber("0.1868+0.04406i")};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{0, 4, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[]{0.84112, 0.32153, 0.78654};

        expShape = new Shape(3, 5);
        expEntries = new CNumber[]{new CNumber("0.84112"), new CNumber("0.34883+0.37796i"), new CNumber("0.32153"), new CNumber("0.78654")};
        expRowIndices = new int[]{0, 0, 1, 2};
        expColIndices = new int[]{0, 4, 0, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new CNumber[]{new CNumber("0.16561+0.20025i"), new CNumber("0.66436+0.01974i"), new CNumber("0.20438+0.61947i"), new CNumber("0.19954+0.15505i"), new CNumber("0.63376+0.2315i")};
        aRowIndices = new int[]{0, 2, 4, 7, 10};
        aColIndices = new int[]{2, 11, 19, 1, 9};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[]{0.60093, 0.98366, 0.43724, 0.76763, 0.1725, 0.96824, 0.24497, 0.75117, 0.20462, 0.34156, 0.01407};

        expShape = new Shape(11, 23);
        expEntries = new CNumber[]{new CNumber("0.16561+0.20025i"), new CNumber("0.60093"), new CNumber("0.98366"), new CNumber("0.43724"), new CNumber("0.66436+0.01974i"), new CNumber("0.76763"), new CNumber("0.1725"), new CNumber("0.20438+0.61947i"), new CNumber("0.96824"), new CNumber("0.24497"), new CNumber("0.19954+0.15505i"), new CNumber("0.75117"), new CNumber("0.20462"), new CNumber("0.34156"), new CNumber("0.01407"), new CNumber("0.63376+0.2315i")};
        expRowIndices = new int[]{0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10};
        expColIndices = new int[]{2, 6, 6, 6, 11, 6, 6, 19, 6, 6, 1, 6, 6, 6, 6, 9};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 6));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new CNumber[]{new CNumber("0.5853+0.62578i"), new CNumber("0.50697+0.75628i"), new CNumber("0.84811+0.20915i"), new CNumber("0.93152+0.99842i"), new CNumber("0.0665+0.22593i"), new CNumber("0.93453+0.96588i"), new CNumber("0.01539+0.54251i"), new CNumber("0.85093+0.09346i"), new CNumber("0.91699+0.88117i"), new CNumber("0.1696+0.62895i"), new CNumber("0.31437+0.90441i"), new CNumber("0.10218+0.59149i"), new CNumber("0.35725+0.49544i"), new CNumber("0.23538+0.03229i"), new CNumber("0.32467+0.04537i"), new CNumber("0.50681+0.72314i"), new CNumber("0.6087+0.14433i"), new CNumber("0.18423+0.43425i"), new CNumber("0.22801+0.90201i"), new CNumber("0.01093+0.89965i"), new CNumber("0.08851+0.37156i"), new CNumber("0.4066+0.33713i"), new CNumber("0.52417+0.70815i"), new CNumber("0.44384+0.05098i"), new CNumber("0.67842+0.76185i"), new CNumber("0.23504+0.86886i"), new CNumber("0.48133+0.76004i"), new CNumber("0.17917+0.58831i"), new CNumber("0.99041+0.8416i"), new CNumber("0.01503+0.1355i"), new CNumber("0.61365+0.20576i"), new CNumber("0.80079+0.27425i"), new CNumber("0.41644+0.50425i"), new CNumber("0.86409+0.61874i"), new CNumber("0.60162+0.39361i"), new CNumber("0.91402+0.63571i"), new CNumber("0.94501+0.85402i"), new CNumber("0.71527+0.00012i"), new CNumber("0.33901+0.57643i"), new CNumber("0.30789+0.4408i"), new CNumber("0.15554+0.86412i"), new CNumber("0.75442+0.68628i"), new CNumber("0.67034+0.64272i"), new CNumber("0.24421+0.81385i"), new CNumber("0.52792+0.8983i"), new CNumber("0.89875+0.81942i"), new CNumber("0.99909+0.31637i"), new CNumber("0.46211+0.99158i"), new CNumber("0.95925+0.85556i"), new CNumber("0.06952+0.10368i"), new CNumber("0.32045+0.55046i"), new CNumber("0.40735+0.274i"), new CNumber("0.95202+0.53482i"), new CNumber("0.15382+0.62718i"), new CNumber("0.57917+0.10623i"), new CNumber("0.12871+0.56135i"), new CNumber("0.35705+0.37999i"), new CNumber("0.7001+0.58944i"), new CNumber("0.24774+0.82336i"), new CNumber("0.56627+0.52135i"), new CNumber("0.23753+0.60904i"), new CNumber("0.42591+0.16206i"), new CNumber("0.32441+0.49509i"), new CNumber("0.48891+0.72823i"), new CNumber("0.7201+0.99531i"), new CNumber("0.65618+0.29415i"), new CNumber("0.54713+0.24437i"), new CNumber("0.35783+0.24487i"), new CNumber("0.49489+0.48034i"), new CNumber("0.67892+0.00546i"), new CNumber("0.77749+0.27982i"), new CNumber("0.3606+0.57591i"), new CNumber("0.93923+0.6653i"), new CNumber("0.68442+0.63347i"), new CNumber("0.34025+0.98589i"), new CNumber("0.8753+0.8315i"), new CNumber("0.32421+0.77341i"), new CNumber("0.65851+0.04788i"), new CNumber("0.20292+0.96589i"), new CNumber("0.69531+0.19398i"), new CNumber("0.07246+0.2818i"), new CNumber("0.7476+0.7582i"), new CNumber("0.41085+0.03654i"), new CNumber("0.74022+0.28372i"), new CNumber("0.64523+0.81809i"), new CNumber("0.06038+0.8067i"), new CNumber("0.11769+0.39042i"), new CNumber("0.84722+0.96926i"), new CNumber("0.90799+0.20162i"), new CNumber("0.53491+0.47647i"), new CNumber("0.68312+0.15704i"), new CNumber("0.84516+0.20208i"), new CNumber("0.16801+0.11932i"), new CNumber("0.66721+0.17513i"), new CNumber("0.5511+0.61789i"), new CNumber("0.11203+0.74791i"), new CNumber("0.10155+0.16813i"), new CNumber("0.32402+0.95958i")};
        aRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        aColIndices = new int[]{93, 110, 143, 147, 149, 169, 301, 372, 485, 523, 584, 630, 645, 711, 813, 885, 898, 85, 157, 241, 324, 332, 353, 391, 482, 501, 548, 579, 584, 605, 771, 926, 37, 68, 83, 102, 222, 266, 280, 300, 323, 376, 472, 487, 560, 604, 623, 657, 701, 751, 819, 851, 927, 968, 14, 56, 64, 93, 112, 141, 200, 205, 208, 213, 250, 320, 335, 378, 409, 466, 475, 480, 550, 557, 594, 606, 746, 762, 766, 776, 813, 909, 202, 254, 297, 302, 313, 406, 423, 514, 629, 638, 712, 735, 759, 860, 988, 996};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[]{0.61356, 0.05473, 0.0305, 0.30243, 0.2978};

        expShape = new Shape(5, 1000);
        expEntries = new CNumber[]{new CNumber("0.5853+0.62578i"), new CNumber("0.50697+0.75628i"), new CNumber("0.84811+0.20915i"), new CNumber("0.93152+0.99842i"), new CNumber("0.0665+0.22593i"), new CNumber("0.93453+0.96588i"), new CNumber("0.01539+0.54251i"), new CNumber("0.85093+0.09346i"), new CNumber("0.91699+0.88117i"), new CNumber("0.1696+0.62895i"), new CNumber("0.31437+0.90441i"), new CNumber("0.10218+0.59149i"), new CNumber("0.35725+0.49544i"), new CNumber("0.23538+0.03229i"), new CNumber("0.32467+0.04537i"), new CNumber("0.50681+0.72314i"), new CNumber("0.6087+0.14433i"), new CNumber("0.61356"), new CNumber("0.18423+0.43425i"), new CNumber("0.22801+0.90201i"), new CNumber("0.01093+0.89965i"), new CNumber("0.08851+0.37156i"), new CNumber("0.4066+0.33713i"), new CNumber("0.52417+0.70815i"), new CNumber("0.44384+0.05098i"), new CNumber("0.67842+0.76185i"), new CNumber("0.23504+0.86886i"), new CNumber("0.48133+0.76004i"), new CNumber("0.17917+0.58831i"), new CNumber("0.99041+0.8416i"), new CNumber("0.01503+0.1355i"), new CNumber("0.61365+0.20576i"), new CNumber("0.80079+0.27425i"), new CNumber("0.05473"), new CNumber("0.41644+0.50425i"), new CNumber("0.86409+0.61874i"), new CNumber("0.60162+0.39361i"), new CNumber("0.91402+0.63571i"), new CNumber("0.94501+0.85402i"), new CNumber("0.71527+0.00012i"), new CNumber("0.33901+0.57643i"), new CNumber("0.30789+0.4408i"), new CNumber("0.15554+0.86412i"), new CNumber("0.75442+0.68628i"), new CNumber("0.67034+0.64272i"), new CNumber("0.24421+0.81385i"), new CNumber("0.52792+0.8983i"), new CNumber("0.89875+0.81942i"), new CNumber("0.99909+0.31637i"), new CNumber("0.46211+0.99158i"), new CNumber("0.95925+0.85556i"), new CNumber("0.06952+0.10368i"), new CNumber("0.32045+0.55046i"), new CNumber("0.40735+0.274i"), new CNumber("0.95202+0.53482i"), new CNumber("0.15382+0.62718i"), new CNumber("0.0305"), new CNumber("0.57917+0.10623i"), new CNumber("0.12871+0.56135i"), new CNumber("0.35705+0.37999i"), new CNumber("0.7001+0.58944i"), new CNumber("0.24774+0.82336i"), new CNumber("0.56627+0.52135i"), new CNumber("0.23753+0.60904i"), new CNumber("0.42591+0.16206i"), new CNumber("0.32441+0.49509i"), new CNumber("0.48891+0.72823i"), new CNumber("0.7201+0.99531i"), new CNumber("0.65618+0.29415i"), new CNumber("0.54713+0.24437i"), new CNumber("0.35783+0.24487i"), new CNumber("0.49489+0.48034i"), new CNumber("0.67892+0.00546i"), new CNumber("0.77749+0.27982i"), new CNumber("0.3606+0.57591i"), new CNumber("0.93923+0.6653i"), new CNumber("0.68442+0.63347i"), new CNumber("0.34025+0.98589i"), new CNumber("0.8753+0.8315i"), new CNumber("0.32421+0.77341i"), new CNumber("0.65851+0.04788i"), new CNumber("0.20292+0.96589i"), new CNumber("0.69531+0.19398i"), new CNumber("0.07246+0.2818i"), new CNumber("0.7476+0.7582i"), new CNumber("0.30243"), new CNumber("0.41085+0.03654i"), new CNumber("0.74022+0.28372i"), new CNumber("0.64523+0.81809i"), new CNumber("0.06038+0.8067i"), new CNumber("0.11769+0.39042i"), new CNumber("0.84722+0.96926i"), new CNumber("0.90799+0.20162i"), new CNumber("0.53491+0.47647i"), new CNumber("0.68312+0.15704i"), new CNumber("0.84516+0.20208i"), new CNumber("0.16801+0.11932i"), new CNumber("0.66721+0.17513i"), new CNumber("0.5511+0.61789i"), new CNumber("0.11203+0.74791i"), new CNumber("0.10155+0.16813i"), new CNumber("0.32402+0.95958i"), new CNumber("0.2978")};
        expRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        expColIndices = new int[]{93, 110, 143, 147, 149, 169, 301, 372, 485, 523, 584, 630, 645, 711, 813, 885, 898, 999, 85, 157, 241, 324, 332, 353, 391, 482, 501, 548, 579, 584, 605, 771, 926, 999, 37, 68, 83, 102, 222, 266, 280, 300, 323, 376, 472, 487, 560, 604, 623, 657, 701, 751, 819, 851, 927, 968, 999, 14, 56, 64, 93, 112, 141, 200, 205, 208, 213, 250, 320, 335, 378, 409, 466, 475, 480, 550, 557, 594, 606, 746, 762, 766, 776, 813, 909, 999, 202, 254, 297, 302, 313, 406, 423, 514, 629, 638, 712, 735, 759, 860, 988, 996, 999};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 999));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.9702+0.73491i"), new CNumber("0.46628+0.86554i"), new CNumber("0.4483+0.41039i"), new CNumber("0.17044+0.60842i")};
        aRowIndices = new int[]{0, 1, 2, 4};
        aColIndices = new int[]{1, 0, 1, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[]{0.19505, 0.25381, 0.50734, 0.63465, 0.06521};

        CooCMatrix final0a = a;
        double[] final0b = b;
        assertThrows(Exception.class, ()->final0a.setCol(final0b, -1));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.94116+0.09216i"), new CNumber("0.14999+0.1742i"), new CNumber("0.71393+0.18301i"), new CNumber("0.64386+0.47526i")};
        aRowIndices = new int[]{0, 2, 2, 4};
        aColIndices = new int[]{2, 0, 1, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[]{0.87499, 0.0504, 0.78606, 0.26238, 0.90258};

        CooCMatrix final1a = a;
        double[] final1b = b;
        assertThrows(Exception.class, ()->final1a.setCol(final1b, 4));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.52339+0.31481i"), new CNumber("0.26671+0.97591i"), new CNumber("0.39437+0.97338i"), new CNumber("0.79979+0.79267i")};
        aRowIndices = new int[]{0, 1, 2, 3};
        aColIndices = new int[]{1, 1, 2, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[]{0.14314, 0.09869, 0.39114, 0.17104, 0.5663, 0.49907, 0.33762};

        CooCMatrix final2a = a;
        double[] final2b = b;
        assertThrows(Exception.class, ()->final2a.setCol(final2b, 0));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.26473+0.15072i"), new CNumber("0.9741+0.31421i"), new CNumber("0.87629+0.74851i"), new CNumber("0.74989+0.9584i")};
        aRowIndices = new int[]{0, 2, 2, 4};
        aColIndices = new int[]{0, 1, 2, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[]{0.98713, 0.85368, 0.61487};

        CooCMatrix final3a = a;
        double[] final3b = b;
        assertThrows(Exception.class, ()->final3a.setCol(final3b, 0));
    }

}
