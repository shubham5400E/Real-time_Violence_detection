import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { CameraStream } from './CameraStream';
import { format } from 'date-fns';
import { api } from '../lib/api';

interface Notification {
  id: string;
  notification_text: string;
  timestamp: string;
  camera_id: string;
}

interface Camera {
  id: string;
  camera_name: string;
  ip_address: string;
  status: string;
}

export function CameraDetails() {
  const { id } = useParams<{ id: string }>();
  const [camera, setCamera] = useState<Camera | null>(null);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [detectionStatus, setDetectionStatus] = useState<string>('stopped');

  useEffect(() => {
    fetchCameraAndNotifications();

    const channel = supabase
      .channel(`camera-${id}`)
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'notifications',
          filter: `camera_id=eq.${id}`,
        },
        (payload) => {
          setNotifications((current) => [payload.new as Notification, ...current]);
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [id]);

  const fetchCameraAndNotifications = async () => {
    try {
      const cameraResult = await supabase.from('cameras').select('*').eq('id', id).single();
      const notificationsResult = await supabase
        .from('notifications')
        .select('*')
        .eq('camera_id', id)
        .order('timestamp', { ascending: false });

      if (cameraResult.error) throw cameraResult.error;
      if (notificationsResult.error) throw notificationsResult.error;

      setCamera(cameraResult.data);
      setNotifications(notificationsResult.data || []);
      setDetectionStatus(cameraResult.data.status); // ✅ Set initial detection status
    } catch (error: any) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const toggleDetection = async () => {
    try {
      const { data, error } = await supabase
        .from('cameras')
        .select('status')
        .eq('id', id)
        .single();

      if (error) throw new Error(error.message);
      if (!data) throw new Error('Camera not found');

      const newStatus = data.status === 'stopped' ? 'started' : 'stopped';

      const { error: updateError } = await supabase
        .from('cameras')
        .update({ status: newStatus })
        .eq('id', id);

      if (updateError) throw new Error(updateError.message);

      api.update_detection_status(newStatus, id);

      setDetectionStatus(newStatus); // ✅ Update detection status in the state
    } catch (err: any) {
      setError(err.message);
    }
  };

  if (loading) return <div className="text-center p-4 text-white">Loading...</div>;
  if (error) return <div className="text-red-500 p-4 text-center">{error}</div>;
  if (!camera) return <div className="text-center p-4 text-white">Camera not found</div>;

  return (
    <div className="min-h-screen p-6 bg-gradient-to-br from-[#141e30] to-[#243b55]">
      <div className="glass rounded-xl p-6 space-y-8 max-w-4xl mx-auto">
        <h2 className="text-3xl font-bold text-white">{camera.camera_name}</h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="glass rounded-xl p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Live Feed</h3>
            <div className="aspect-video rounded-lg overflow-hidden border border-white/10">
              <CameraStream ipAddress={camera.ip_address} cameraName={camera.camera_name} />
            </div>
          </div>

          <div className="glass rounded-xl p-6 max-h-[600px] overflow-y-auto">
            <h3 className="text-xl font-semibold text-white mb-4">Notifications</h3>
            {notifications.length > 0 ? (
              notifications.map((notification) => (
                <div
                  key={notification.id}
                  className="p-4 glass rounded-lg mb-4 text-white/90 border border-white/20"
                >
                  <p className="font-medium">{notification.notification_text}</p>
                  <p className="text-sm text-white/70 mt-1">
                    {format(new Date(notification.timestamp), 'PPp')}
                  </p>
                </div>
              ))
            ) : (
              <p className="text-white/60 text-center py-4">No notifications yet</p>
            )}
          </div>
        </div>

        <div className="glass rounded-xl p-6 text-center">
          <p className="text-xl font-semibold text-white mb-4">
            Detection Status: {detectionStatus === 'stopped' ? 'Detection stopped' : 'Detection started'}
          </p>
          <button
            onClick={toggleDetection}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
          >
            {detectionStatus === 'stopped' ? 'Start Detection' : 'Stop Detection'}
          </button>
        </div>
      </div>
    </div>
  );
}
