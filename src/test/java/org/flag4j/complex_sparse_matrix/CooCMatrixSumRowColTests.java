package org.flag4j.complex_sparse_matrix;

import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CooCMatrixSumRowColTests {


    @Test
    void sumRowsTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.75984+0.23557i"), new CNumber("0.93444+0.78486i"), new CNumber("0.0436+0.95139i")};
        aRowIndices = new int[]{1, 1, 2};
        aColIndices = new int[]{1, 3, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1, 5);
        expEntries = new CNumber[]{new CNumber("0.75984+0.23557i"), new CNumber("0.0436+0.95139i"), new CNumber("0.93444+0.78486i")};
        expRowIndices = new int[]{0, 0, 0};
        expColIndices = new int[]{1, 2, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp.toDense().toVector(), a.sumRows());

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new CNumber[]{new CNumber("0.18199+0.68543i"), new CNumber("0.23849+0.77354i"), new CNumber("0.09727+0.59895i"), new CNumber("0.06741+0.84726i"), new CNumber("0.33347+0.0043i")};
        aRowIndices = new int[]{0, 2, 3, 7, 10};
        aColIndices = new int[]{4, 3, 7, 21, 16};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1, 23);
        expEntries = new CNumber[]{new CNumber("0.23849+0.77354i"), new CNumber("0.18199+0.68543i"), new CNumber("0.09727+0.59895i"), new CNumber("0.33347+0.0043i"), new CNumber("0.06741+0.84726i")};
        expRowIndices = new int[]{0, 0, 0, 0, 0};
        expColIndices = new int[]{3, 4, 7, 16, 21};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp.toDense().toVector(), a.sumRows());

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new CNumber[]{new CNumber("0.63353+0.77172i"), new CNumber("0.25092+0.74284i"), new CNumber("0.31761+0.92992i"), new CNumber("0.58917+0.46272i"), new CNumber("0.76358+0.9196i"), new CNumber("0.42157+0.97527i"), new CNumber("0.18876+0.8494i"), new CNumber("0.00812+0.8152i"), new CNumber("0.91001+0.69971i"), new CNumber("0.77041+0.37788i"), new CNumber("0.52607+0.33079i"), new CNumber("0.97761+0.34024i"), new CNumber("0.02372+0.79217i"), new CNumber("0.16756+0.10463i"), new CNumber("0.77036+0.95128i"), new CNumber("0.61751+0.26009i"), new CNumber("0.46849+0.57104i"), new CNumber("0.52918+0.44938i"), new CNumber("0.59969+0.65922i"), new CNumber("0.47107+0.8591i"), new CNumber("0.49107+0.78737i"), new CNumber("0.75273+0.70286i"), new CNumber("0.53593+0.24363i"), new CNumber("0.23275+0.36013i"), new CNumber("0.88056+0.14132i"), new CNumber("0.31367+0.9771i"), new CNumber("0.18406+0.11809i"), new CNumber("0.54932+0.55517i"), new CNumber("0.02436+0.86583i"), new CNumber("0.32834+0.9869i"), new CNumber("0.88876+0.60651i"), new CNumber("0.58743+0.99918i"), new CNumber("0.23772+0.22926i"), new CNumber("0.09036+0.9595i"), new CNumber("0.12409+0.11868i"), new CNumber("0.67328+0.70908i"), new CNumber("0.98855+0.11612i"), new CNumber("0.57897+0.07066i"), new CNumber("0.70447+0.18972i"), new CNumber("0.67642+0.16493i"), new CNumber("0.71806+0.50502i"), new CNumber("0.2421+0.41935i"), new CNumber("0.51417+0.74713i"), new CNumber("0.96518+0.95548i"), new CNumber("0.64315+0.71817i"), new CNumber("0.83589+0.7442i"), new CNumber("0.67351+0.67196i"), new CNumber("0.72576+0.45976i"), new CNumber("0.92805+0.16953i"), new CNumber("0.45938+0.78156i"), new CNumber("0.4914+0.45498i"), new CNumber("0.91129+0.79717i"), new CNumber("0.53956+0.09054i"), new CNumber("0.83644+0.62089i"), new CNumber("0.17713+0.29427i"), new CNumber("0.88846+0.25136i"), new CNumber("0.57263+0.63916i"), new CNumber("0.0649+0.97677i"), new CNumber("0.03396+0.03522i"), new CNumber("0.76154+0.04389i"), new CNumber("0.4285+0.3314i"), new CNumber("0.2454+0.88584i"), new CNumber("0.75813+0.6582i"), new CNumber("0.12882+0.37636i"), new CNumber("0.53808+0.91221i"), new CNumber("0.80563+0.18208i"), new CNumber("0.92844+0.03075i"), new CNumber("0.19168+0.73507i"), new CNumber("0.84598+0.69869i"), new CNumber("0.26348+0.28796i"), new CNumber("0.67996+0.66392i"), new CNumber("0.28063+0.2548i"), new CNumber("0.75967+0.64006i"), new CNumber("0.04529+0.76163i"), new CNumber("0.43085+0.40973i"), new CNumber("0.53847+0.40672i"), new CNumber("0.96204+0.02381i"), new CNumber("0.18778+0.83512i"), new CNumber("0.17525+0.77112i"), new CNumber("0.71284+0.95122i"), new CNumber("0.22359+0.62776i"), new CNumber("0.33849+0.10369i"), new CNumber("0.16256+0.10552i"), new CNumber("0.41425+0.9014i"), new CNumber("0.06284+0.99073i"), new CNumber("0.86131+0.26804i"), new CNumber("0.53608+0.95183i"), new CNumber("0.18228+0.69169i"), new CNumber("0.65407+0.50937i"), new CNumber("0.78767+0.67874i"), new CNumber("0.62271+0.46654i"), new CNumber("0.41166+0.6821i"), new CNumber("0.74932+0.98614i"), new CNumber("0.01523+0.84099i"), new CNumber("0.6471+0.37479i"), new CNumber("0.99476+0.5966i"), new CNumber("0.48093+0.64855i"), new CNumber("0.23364+0.02951i")};
        aRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        aColIndices = new int[]{101, 104, 197, 274, 303, 304, 373, 427, 434, 505, 521, 542, 546, 583, 625, 689, 837, 853, 110, 119, 146, 147, 186, 504, 577, 628, 703, 710, 720, 800, 870, 924, 962, 970, 996, 154, 162, 219, 316, 352, 466, 521, 533, 584, 607, 611, 646, 751, 833, 855, 877, 899, 25, 37, 51, 91, 121, 140, 142, 200, 233, 290, 379, 401, 419, 447, 468, 495, 556, 559, 662, 871, 885, 923, 984, 33, 84, 102, 120, 160, 308, 325, 337, 353, 356, 436, 444, 483, 535, 537, 617, 655, 718, 808, 845, 926, 978, 987};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1, 1000);
        expEntries = new CNumber[]{new CNumber("0.53956+0.09054i"), new CNumber("0.53847+0.40672i"), new CNumber("0.83644+0.62089i"), new CNumber("0.17713+0.29427i"), new CNumber("0.96204+0.02381i"), new CNumber("0.88846+0.25136i"), new CNumber("0.63353+0.77172i"), new CNumber("0.18778+0.83512i"), new CNumber("0.25092+0.74284i"), new CNumber("0.59969+0.65922i"), new CNumber("0.47107+0.8591i"), new CNumber("0.17525+0.77112i"), new CNumber("0.57263+0.63916i"), new CNumber("0.0649+0.97677i"), new CNumber("0.03396+0.03522i"), new CNumber("0.49107+0.78737i"), new CNumber("0.75273+0.70286i"), new CNumber("0.67328+0.70908i"), new CNumber("0.71284+0.95122i"), new CNumber("0.98855+0.11612i"), new CNumber("0.53593+0.24363i"), new CNumber("0.31761+0.92992i"), new CNumber("0.76154+0.04389i"), new CNumber("0.57897+0.07066i"), new CNumber("0.4285+0.3314i"), new CNumber("0.58917+0.46272i"), new CNumber("0.2454+0.88584i"), new CNumber("0.76358+0.9196i"), new CNumber("0.42157+0.97527i"), new CNumber("0.22359+0.62776i"), new CNumber("0.70447+0.18972i"), new CNumber("0.33849+0.10369i"), new CNumber("0.16256+0.10552i"), new CNumber("0.67642+0.16493i"), new CNumber("0.41425+0.9014i"), new CNumber("0.06284+0.99073i"), new CNumber("0.18876+0.8494i"), new CNumber("0.75813+0.6582i"), new CNumber("0.12882+0.37636i"), new CNumber("0.53808+0.91221i"), new CNumber("0.00812+0.8152i"), new CNumber("0.91001+0.69971i"), new CNumber("0.86131+0.26804i"), new CNumber("0.53608+0.95183i"), new CNumber("0.80563+0.18208i"), new CNumber("0.71806+0.50502i"), new CNumber("0.92844+0.03075i"), new CNumber("0.18228+0.69169i"), new CNumber("0.19168+0.73507i"), new CNumber("0.23275+0.36013i"), new CNumber("0.77041+0.37788i"), new CNumber("0.76817+0.75014i"), new CNumber("0.51417+0.74713i"), new CNumber("0.65407+0.50937i"), new CNumber("0.78767+0.67874i"), new CNumber("0.97761+0.34024i"), new CNumber("0.02372+0.79217i"), new CNumber("0.84598+0.69869i"), new CNumber("0.26348+0.28796i"), new CNumber("0.88056+0.14132i"), new CNumber("0.16756+0.10463i"), new CNumber("0.96518+0.95548i"), new CNumber("0.64315+0.71817i"), new CNumber("0.83589+0.7442i"), new CNumber("0.62271+0.46654i"), new CNumber("0.77036+0.95128i"), new CNumber("0.31367+0.9771i"), new CNumber("0.67351+0.67196i"), new CNumber("0.41166+0.6821i"), new CNumber("0.67996+0.66392i"), new CNumber("0.61751+0.26009i"), new CNumber("0.18406+0.11809i"), new CNumber("0.54932+0.55517i"), new CNumber("0.74932+0.98614i"), new CNumber("0.02436+0.86583i"), new CNumber("0.72576+0.45976i"), new CNumber("0.32834+0.9869i"), new CNumber("0.01523+0.84099i"), new CNumber("0.92805+0.16953i"), new CNumber("0.46849+0.57104i"), new CNumber("0.6471+0.37479i"), new CNumber("0.52918+0.44938i"), new CNumber("0.45938+0.78156i"), new CNumber("0.88876+0.60651i"), new CNumber("0.28063+0.2548i"), new CNumber("0.4914+0.45498i"), new CNumber("0.75967+0.64006i"), new CNumber("0.91129+0.79717i"), new CNumber("0.04529+0.76163i"), new CNumber("0.58743+0.99918i"), new CNumber("0.99476+0.5966i"), new CNumber("0.23772+0.22926i"), new CNumber("0.09036+0.9595i"), new CNumber("0.48093+0.64855i"), new CNumber("0.43085+0.40973i"), new CNumber("0.23364+0.02951i"), new CNumber("0.12409+0.11868i")};
        expRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        expColIndices = new int[]{25, 33, 37, 51, 84, 91, 101, 102, 104, 110, 119, 120, 121, 140, 142, 146, 147, 154, 160, 162, 186, 197, 200, 219, 233, 274, 290, 303, 304, 308, 316, 325, 337, 352, 353, 356, 373, 379, 401, 419, 427, 434, 436, 444, 447, 466, 468, 483, 495, 504, 505, 521, 533, 535, 537, 542, 546, 556, 559, 577, 583, 584, 607, 611, 617, 625, 628, 646, 655, 662, 689, 703, 710, 718, 720, 751, 800, 808, 833, 837, 845, 853, 855, 870, 871, 877, 885, 899, 923, 924, 926, 962, 970, 978, 984, 987, 996};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp.toDense().toVector(), a.sumRows());

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.84356+0.58055i"), new CNumber("0.15836+0.25564i"), new CNumber("0.39453+0.02289i"), new CNumber("0.47833+0.94144i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{0, 0, 1, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1, 3);
        expEntries = new CNumber[]{new CNumber("1.00192+0.83619i"), new CNumber("0.87286+0.96433i")};
        expRowIndices = new int[]{0, 0};
        expColIndices = new int[]{0, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp.toDense().toVector(), a.sumRows());

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.74152+0.29525i"), new CNumber("0.11269+0.78219i"), new CNumber("0.52011+0.84505i"), new CNumber("0.22752+0.62699i")};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{1, 1, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1, 3);
        expEntries = new CNumber[]{new CNumber("1.37432+1.9224900000000003i"), new CNumber("0.22752+0.62699i")};
        expRowIndices = new int[]{0, 0};
        expColIndices = new int[]{1, 2};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp.toDense().toVector(), a.sumRows());

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.49817+0.09288i"), new CNumber("0.56148+0.20833i"), new CNumber("0.87569+0.85507i"), new CNumber("0.40512+0.01566i")};
        aRowIndices = new int[]{0, 1, 3, 4};
        aColIndices = new int[]{1, 0, 1, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1, 3);
        expEntries = new CNumber[]{new CNumber("0.9665999999999999+0.22399i"), new CNumber("1.37386+0.94795i")};
        expRowIndices = new int[]{0, 0};
        expColIndices = new int[]{0, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp.toDense().toVector(), a.sumRows());

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.74205+0.36823i"), new CNumber("0.06271+0.64269i"), new CNumber("0.16997+0.08338i"), new CNumber("0.5575+0.61039i")};
        aRowIndices = new int[]{0, 2, 3, 4};
        aColIndices = new int[]{2, 1, 0, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1, 3);
        expEntries = new CNumber[]{new CNumber("0.16997+0.08338i"), new CNumber("0.06271+0.64269i"), new CNumber("1.29955+0.97862i")};
        expRowIndices = new int[]{0, 0, 0};
        expColIndices = new int[]{0, 1, 2};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp.toDense().toVector(), a.sumRows());
    }


    @Test
    void sumColsTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.71374+0.30282i"), new CNumber("0.13214+0.97552i"), new CNumber("0.70794+0.6353i")};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{2, 2, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3, 1);
        expEntries = new CNumber[]{new CNumber("0.71374+0.30282i"), new CNumber("0.13214+0.97552i"), new CNumber("0.70794+0.6353i")};
        expRowIndices = new int[]{0, 1, 2};
        expColIndices = new int[]{0, 0, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp.toDense().toVector(), a.sumCols());

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new CNumber[]{new CNumber("0.28302+0.05646i"), new CNumber("0.07482+0.05769i"), new CNumber("0.26845+0.8736i"), new CNumber("0.20683+0.56873i"), new CNumber("0.47758+0.83312i")};
        aRowIndices = new int[]{0, 7, 8, 8, 9};
        aColIndices = new int[]{8, 7, 16, 22, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(11, 1);
        expEntries = new CNumber[]{new CNumber("0.28302+0.05646i"), new CNumber("0.07482+0.05769i"), new CNumber("0.47528000000000004+1.4423300000000001i"), new CNumber("0.47758+0.83312i")};
        expRowIndices = new int[]{0, 7, 8, 9};
        expColIndices = new int[]{0, 0, 0, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp.toDense().toVector(), a.sumCols());

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new CNumber[]{new CNumber("0.65595+0.49703i"), new CNumber("0.06541+0.0893i"), new CNumber("0.87203+0.49868i"), new CNumber("0.14488+0.61777i"), new CNumber("0.06557+0.27304i"), new CNumber("0.51688+0.50355i"), new CNumber("0.6222+0.5684i"), new CNumber("0.7522+0.34201i"), new CNumber("0.58039+0.24547i"), new CNumber("0.32789+0.93833i"), new CNumber("0.58593+0.66788i"), new CNumber("0.50668+0.56456i"), new CNumber("0.21896+0.57989i"), new CNumber("0.02268+0.7085i"), new CNumber("0.44206+0.11124i"), new CNumber("0.84889+0.86584i"), new CNumber("0.32971+0.819i"), new CNumber("0.05307+0.13348i"), new CNumber("0.39368+0.7446i"), new CNumber("0.1199+0.5361i"), new CNumber("0.12211+0.50908i"), new CNumber("0.3502+0.88431i"), new CNumber("0.49695+0.73371i"), new CNumber("0.14239+0.17296i"), new CNumber("0.76284+0.16894i"), new CNumber("0.33393+0.85186i"), new CNumber("0.5748+0.25347i"), new CNumber("0.18947+0.41814i"), new CNumber("0.81342+0.46603i"), new CNumber("0.454+0.99116i"), new CNumber("0.67078+0.73175i"), new CNumber("0.44353+0.32696i"), new CNumber("0.30019+0.64741i"), new CNumber("0.10412+0.31778i"), new CNumber("0.55985+0.69539i"), new CNumber("0.2233+0.16016i"), new CNumber("0.47102+0.89489i"), new CNumber("0.08427+0.55157i"), new CNumber("0.25409+0.08687i"), new CNumber("0.83413+0.09808i"), new CNumber("0.10613+0.49688i"), new CNumber("0.24048+0.90827i"), new CNumber("0.89422+0.51965i"), new CNumber("0.70325+0.39583i"), new CNumber("0.28297+0.90254i"), new CNumber("0.38004+0.27198i"), new CNumber("0.38558+0.8722i"), new CNumber("0.20746+0.87307i"), new CNumber("0.81842+0.30475i"), new CNumber("0.74157+0.21328i"), new CNumber("0.11761+0.29384i"), new CNumber("0.11055+0.97811i"), new CNumber("0.75044+0.62813i"), new CNumber("0.33265+0.06255i"), new CNumber("0.43746+0.95889i"), new CNumber("0.94575+0.87517i"), new CNumber("0.85749+0.59834i"), new CNumber("0.26431+0.70432i"), new CNumber("0.38943+0.85478i"), new CNumber("0.99204+0.90029i"), new CNumber("0.5396+0.94499i"), new CNumber("0.21098+0.76004i"), new CNumber("0.82396+0.83141i"), new CNumber("0.87743+0.4428i"), new CNumber("0.69052+0.84171i"), new CNumber("0.17258+0.6132i"), new CNumber("0.69727+0.62668i"), new CNumber("0.84969+0.000006i"), new CNumber("0.48049+0.88647i"), new CNumber("0.63826+0.31349i"), new CNumber("0.46923+0.6246i"), new CNumber("0.93621+0.34855i"), new CNumber("0.03882+0.95189i"), new CNumber("0.41115+0.62505i"), new CNumber("0.29721+0.03672i"), new CNumber("0.38713+0.65758i"), new CNumber("0.96719+0.41359i"), new CNumber("0.55215+0.73225i"), new CNumber("0.84709+0.29944i"), new CNumber("0.65176+0.45494i"), new CNumber("0.32923+0.31776i"), new CNumber("0.61+0.89127i"), new CNumber("0.37462+0.21452i"), new CNumber("0.85101+0.47374i"), new CNumber("0.70642+0.81323i"), new CNumber("0.39814+0.52492i"), new CNumber("0.13433+0.39392i"), new CNumber("0.55352+0.0385i"), new CNumber("0.31503+0.95855i"), new CNumber("0.60146+0.90334i"), new CNumber("0.71902+0.60254i"), new CNumber("0.82981+0.31763i"), new CNumber("0.11739+0.90946i"), new CNumber("0.30151+0.01799i"), new CNumber("0.64173+0.42104i"), new CNumber("0.03976+0.56942i"), new CNumber("0.35345+0.81972i"), new CNumber("0.04665+0.11388i")};
        aRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        aColIndices = new int[]{62, 95, 140, 208, 236, 263, 345, 406, 413, 429, 432, 436, 479, 525, 549, 588, 672, 801, 862, 871, 886, 896, 949, 984, 999, 12, 77, 85, 130, 228, 293, 314, 319, 410, 528, 561, 570, 582, 691, 716, 729, 730, 753, 764, 872, 959, 41, 65, 283, 336, 384, 445, 448, 462, 552, 643, 681, 695, 868, 896, 930, 944, 27, 128, 145, 223, 353, 356, 388, 397, 429, 501, 509, 566, 705, 775, 928, 110, 119, 128, 196, 261, 290, 299, 324, 415, 429, 470, 507, 514, 560, 660, 702, 712, 717, 902, 975, 998};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5, 1);
        expEntries = new CNumber[]{new CNumber(9.99945, 12.773669999999997), new CNumber(8.917990000000001, 10.98667), new CNumber(8.10134, 10.822750000000001), new CNumber(8.73714, 8.213745999999999), new CNumber(9.974080000000002, 10.78806)};
        expRowIndices = new int[]{0, 1, 2, 3, 4};
        expColIndices = new int[]{0, 0, 0, 0, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp.toDense().toVector(), a.sumCols());

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.76635+0.93535i"), new CNumber("0.79935+0.2649i"), new CNumber("0.79933+0.21103i"), new CNumber("0.33334+0.95187i")};
        aRowIndices = new int[]{1, 2, 4, 4};
        aColIndices = new int[]{0, 0, 0, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5, 1);
        expEntries = new CNumber[]{new CNumber("0.76635+0.93535i"), new CNumber("0.79935+0.2649i"), new CNumber("1.13267+1.1629i")};
        expRowIndices = new int[]{1, 2, 4};
        expColIndices = new int[]{0, 0, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp.toDense().toVector(), a.sumCols());

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.41027+0.82848i"), new CNumber("0.13789+0.09513i"), new CNumber("0.42385+0.78672i"), new CNumber("0.46963+0.13069i")};
        aRowIndices = new int[]{0, 1, 3, 4};
        aColIndices = new int[]{1, 2, 2, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5, 1);
        expEntries = new CNumber[]{new CNumber("0.41027+0.82848i"), new CNumber("0.13789+0.09513i"), new CNumber("0.42385+0.78672i"), new CNumber("0.46963+0.13069i")};
        expRowIndices = new int[]{0, 1, 3, 4};
        expColIndices = new int[]{0, 0, 0, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp.toDense().toVector(), a.sumCols());

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.87737+0.39295i"), new CNumber("0.84488+0.92298i"), new CNumber("0.50406+0.55808i"), new CNumber("0.96185+0.53169i")};
        aRowIndices = new int[]{0, 1, 2, 4};
        aColIndices = new int[]{0, 0, 2, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5, 1);
        expEntries = new CNumber[]{new CNumber("0.87737+0.39295i"), new CNumber("0.84488+0.92298i"), new CNumber("0.50406+0.55808i"), new CNumber("0.96185+0.53169i")};
        expRowIndices = new int[]{0, 1, 2, 4};
        expColIndices = new int[]{0, 0, 0, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp.toDense().toVector(), a.sumCols());

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.41156+0.43521i"), new CNumber("0.73202+0.08806i"), new CNumber("0.59749+0.51931i"), new CNumber("0.40106+0.40126i")};
        aRowIndices = new int[]{0, 2, 4, 4};
        aColIndices = new int[]{1, 2, 0, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(5, 1);
        expEntries = new CNumber[]{new CNumber("0.41156+0.43521i"), new CNumber("0.73202+0.08806i"), new CNumber("0.99855+0.9205700000000001i")};
        expRowIndices = new int[]{0, 2, 4};
        expColIndices = new int[]{0, 0, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp.toDense().toVector(), a.sumCols());
    }
}
