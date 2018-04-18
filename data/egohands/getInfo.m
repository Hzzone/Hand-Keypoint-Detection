video = getMetaBy();
fid = fopen('egohands_data.txt','w');
for i=1:1:48
    video_id = video(i).video_id;
    for j=1:1:100
        fprintf(fid,'%s ', video_id);
        frame_num = video(i).labelled_frames(j).frame_num;
        fprintf(fid,'%s ', num2str(frame_num));
        boxes = getBoundingBoxes(video(i), j);
        for x=1:4
            if sum(boxes(x, :)) ~=0
                box = boxes(x, :);
                fprintf(fid,'%d %d %d %d ', box(1), box(2), box(3), box(4));
            end
        end
        fprintf(fid,'\n');
    end
end
fclose(fid);

    